Update HaplotypeReader to maintain a sample-chunk cache.

Do not support the ``virtual_ancestor`` haplotype ID here, this edge
case can be handled externally.

Do not make distinctions between the "ancestors" source and other sources.
The should be treated the same internally. Maintain a separate VCZHaplotypeReader
per source, which the HaplotypeReader can provide a facade for per source.

Each VCZHaplotypeReader gets a ``sample_id`` as its input, and returns the
haplotype for that sample on the positions specifed, with the genotypes
polarised so that the ancestral allele is always zero. This requires access
to the ancestral state config.

Internally, the VCFHaplotypeReader maps the ``sample_id`` to its chunk index
and offset within that chunk. It checks the cache for that chunk, and loads
the chunk if there's a miss. The number of sample chunks in the cache is
configurable at run time, and defaults to 3.

**NOTE** Need to reconsider this because it implies we need to keep all the
1000G data in memory at once.
