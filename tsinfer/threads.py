#
# Copyright (C) 2018 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Utilities for handling threads.
"""
import logging
import threading
import traceback
import _thread


# prctl is an optional extra; it allows us assign meaninful names to threads
# for debugging.
_prctl_available = False
try:
    import prctl
    _prctl_available = True
except ImportError:
    pass

_numa_available = False
try:
    import numa
    _numa_available = True
except ImportError:
    pass


logger = logging.getLogger(__name__)


def _queue_thread(worker, work_queue, name="tsinfer-worker", index=0, consumer=True):
    def thread_target():
        try:
            logger.debug("thread '{}' starting".format(name))
            if _prctl_available:
                prctl.set_name(name)
            if _numa_available and numa.available():
                numa.set_localalloc()
                logger.debug(
                    "Set NUMA local allocation policy on thread {}".format(name))
            worker(index)
            logger.debug("thread '{}' finishing".format(name))
        except Exception:
            logger.critical("Exception occured in thread; exiting")
            logger.critical(traceback.format_exc())
            # Communicate back the main thread that something bad has happened.
            # This seems to be the only reliable way to do it.
            _thread.interrupt_main()
            # Now we still need to make sure that the main thread doesn't block
            # on the queue.get/join (as it won't be interrupted). This is an attempt
            # to make sure that it unblocks. May not be fool-proof though.
            #
            # TODO This doesn't really work. We can still block on pushing things
            # onto the queue. We'll probably have to do something ourselves using
            # timeouts and stuff to see if an error has occured.
            if consumer:
                while True:
                    try:
                        work_queue.task_done()
                    except ValueError:
                        break
            else:
                work_queue.put(None)

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def queue_producer_thread(worker, work_queue, name="tsinfer-worker", index=0):
    """
    Returns a started thread that produces items for the specified queue using the
    specified worker function.
    """
    return _queue_thread(worker, work_queue, name=name, index=index, consumer=False)


def queue_consumer_thread(worker, work_queue, name="tsinfer-worker", index=0):
    """
    Returns a started thread that consumes items for the specified queue using the
    specified worker function.
    """
    return _queue_thread(worker, work_queue, name=name, index=index, consumer=True)
