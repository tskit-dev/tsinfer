#
# Copyright (C) 2018-2020 University of Oxford
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
A progress monitor class for tsinfer
"""
from tqdm.auto import tqdm


class ProgressMonitor:
    """
    Class responsible for managing in the tqdm progress monitors.
    """

    def __init__(
        self,
        enabled=True,
        generate_ancestors=False,
        match_ancestors=False,
        augment_ancestors=False,
        match_samples=False,
        verify=False,
        tqdm_kwargs=None,
    ):
        self.enabled = enabled
        self.num_bars = 0
        if generate_ancestors:
            self.num_bars += 2
        if match_ancestors:
            self.num_bars += 1
        if match_samples:
            self.num_bars += 3
        if verify:
            assert self.num_bars == 0
            self.num_bars += 1
        if augment_ancestors:
            assert self.num_bars == 0
            self.num_bars += 2
        self.current_count = 0
        self.current_instance = None
        if not verify:
            # Only show extra detail if we are running match-ancestors by itself.
            self.show_detail = self.num_bars == 1
        self.descriptions = {
            "ga_add_sites": "ga-add",
            "ga_generate": "ga-gen",
            "ma_match": "ma-match",
            "ms_match": "ms-match",
            "ms_paths": "ms-paths",
            "ms_full_mutations": "ms-muts",
            "ms_extra_sites": "ms-xsites",
            "verify": "verify",
        }
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        self.tqdm_kwargs = tqdm_kwargs

    def set_detail(self, info):
        if self.show_detail:
            self.current_instance.set_postfix(info)

    def get(self, key, total):
        self.current_count += 1
        desc = "{:<8} ({}/{})".format(
            self.descriptions[key], self.current_count, self.num_bars
        )
        bar_format = (
            "{desc}{percentage:3.0f}%|{bar}"
            "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
        )
        self.current_instance = tqdm(
            desc=desc,
            total=total,
            disable=not self.enabled,
            bar_format=bar_format,
            dynamic_ncols=True,
            smoothing=0.01,
            unit_scale=True,
            **self.tqdm_kwargs
        )
        return self.current_instance


class DummyProgress:
    """
    Class that mimics the subset of the tqdm API that we use in this module.
    """

    def update(self, n=None):
        pass

    def close(self):
        pass


class DummyProgressMonitor(ProgressMonitor):
    """
    Simple class to mimic the interface of the real progress monitor.
    """

    def get(self, key, total):
        return DummyProgress()

    def set_detail(self, info):
        pass
