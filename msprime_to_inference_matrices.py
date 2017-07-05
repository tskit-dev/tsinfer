import numpy as np

import msprime

def make_ancestral_matrices(ts):
    """
    Take a simulation and reconstruct the ancestors from it
    returns a basic haplotype matrix and a copy matrix, which can
    be further refined by e.g. relabel_copy_matrix
    """
    #make the array - should be a row for each node
    column_mapping = np.zeros(ts.num_sites)
    haplotype_matrix = -np.ones((ts.num_nodes, ts.num_sites), dtype=np.int)
    copying_matrix   = -np.ones((ts.num_nodes, ts.num_sites), dtype=np.int)
    root = np.zeros(ts.num_sites, dtype=np.int)
    mutations = {k:[] for k in range(ts.num_nodes)}
    for v in ts.variants(as_bytes=False):
        column_mapping[v.index] = v.position
        haplotype_matrix[0:ts.sample_size,v.site.index] = v.genotypes
        root[v.site.index] = int(v.site.ancestral_state)
        mutation_state = v.site.ancestral_state
        for m in v.site.mutations:
            # (site, prev state, new state)
            mutations[m.node].append((m.site, mutation_state,m.derived_state))
            mutation_state = m.derived_state
    
    for es in ts.edgesets():
        # these are in youngest -> oldest order, so by iterating in the order given,
        # we should always be able to fill out parents from their children
        mask = (es.left <= column_mapping) & (column_mapping < es.right)
        previous_child = -1
        for child in es.children:
            #fill in the copying matrix
            copying_matrix[child,mask] = es.parent
            #create row for the haplotype matrix
            genotype = -np.ones(ts.num_sites, dtype=np.int)
            genotype[mask] = haplotype_matrix[child,mask]
            # add mutations. These are in oldest -> youngest order, so as we are working from
            # child->parent, we need to reverse
            for site, prev_state, new_state in reversed(mutations[child]):
                if mask[site]: #only set mutations on sites contained in this edgeset
                    assert genotype[site] == int(new_state)
                    genotype[site] = int(prev_state)
            
            if previous_child != -1:
                assert np.array_equal(genotype[mask], haplotype_matrix[es.parent,mask]), \
                    "children {} and {} disagree".format(previous_child, child)
            
            haplotype_matrix[es.parent,mask] = genotype[mask]
            previous_child = child
    
    
    #Check that the ancestral state matches the oldest variant for each column
    for c in range(ts.num_sites):
        col = haplotype_matrix[:,c]
        filled = col[col != -1]
        assert filled[-1] == root[c], \
            "the ancestral state does not match the oldest variant for column {}".format(c)
    return haplotype_matrix, copying_matrix
    
    
def singleton_sites(ts):
    """
    returns:
    a numpy boolean array of length ts.num_sites indicating which sites are singletons
    """
    return np.array([
        np.count_nonzero(v.genotypes != int(v.site.ancestral_state)) <= 1 \
            for v in ts.variants(as_bytes=False) \
        ], dtype=np.bool)

def unused_ancestors(haplotype_matrix):
    #identify nodes that are not used
    #these are cases where the node is used in the Treeseq, but since the
    #genome span does not have any variable sites, it is not filled in the H matrix
    return np.all(haplotype_matrix == -1, 1)

   
def relabel_copy_matrix(copy_matrix, keep_rows):
    """
    Return the copy matrix, reduced by removing certain rows, and with the
    internal numbers relabelled accordingly.
    
    If keep_rows is a numpy boolean vector of length of the number of rows
    then simply remove the other rows and keep the same ordering.
    If keep_rows is a numpy integer vector, it does not need to be the same 
    length, and contains the order of rows to output. This allows each row to 
    correspond to a single mutation, as expected in inference output. However, it also
    may create duplicate rows (where >1 mutation occurs above the same node).
    In this case, any of the duplicate rows could be set as the ancestor. We select
    the  
    """
    def bool_not_in(vec, index):
        return vec[index]==False
    def index_not_in(vec, index):
        return index not in vec
        
    bad_parent_id = -2 #used for refs to rows we have removed (not -1 which is N/A)
    #create an array to map current_row_number (index) -> new row numbers
    node_reposition = {}
    #also as a numpy array, picking the first new row number if there are duplicate rows
    #and bad parents
    node_relabelling = np.full(copy_matrix.shape[0], bad_parent_id, dtype=np.int)
    if keep_rows.dtype == np.bool:
        test_func = bool_not_in
        node_relabelling[keep_rows]=np.arange(np.count_nonzero(keep_rows))
        node_reposition = {[m] for m in node_relabelling  if m!=bad_parent_id}
    else:
        test_func = index_not_in
        for i in range(len(keep_rows)):
            node_reposition.setdefault(keep_rows[i],[]).append(i)
        for k,v in node_reposition.items():
            node_relabelling[k]=v[0]
    #remove unkept rows            
    reduced_matrix = copy_matrix[keep_rows,:]
    #there may be problems here, because we might have removed some rows that
    #are proper ancestors. So for each reference to a row, we should check if it
    #exists in keep_rows, and if not, should follow the ancestry trail until we hit
    #a row that *does* exist
    for i in range(reduced_matrix.shape[0]):
        for j in range(reduced_matrix.shape[1]):
            while test_func(keep_rows, reduced_matrix[i,j]):
                #replace this one
                reduced_matrix[i,j] = copy_matrix[reduced_matrix[i,j],j]
                if reduced_matrix[i,j] == -1:
                    break

    #hack so that node_relabelling[-1] (absent) still maps to -1
    node_relabelling = np.append(node_relabelling, [-1])
    
    relabelled_matrix = node_relabelling[reduced_matrix]
    assert not np.any(relabelled_matrix == bad_parent_id)
    assert not np.any(relabelled_matrix >= relabelled_matrix.shape[0])
    return relabelled_matrix, node_reposition
 