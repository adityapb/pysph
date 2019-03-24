//CL//

// IMPORTANT NOTE: pyopencl uses the length of the first argument
// to determine the global work size

<%def name="preamble()" cached="True">
</%def>


<%def name="fill_pids_args(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, ${data_t} cell_size,
    ${data_t}3 min, unsigned long* keys, unsigned int* pids
</%def>

<%def name="fill_pids_src(data_t)" cached="True">
    unsigned long c_x, c_y, c_z;
    FIND_CELL_ID(
        x[i] - min.x,
        y[i] - min.y,
        z[i] - min.z,
        cell_size, c_x, c_y, c_z
        );
    unsigned long key;
    key = interleave(c_x, c_y, c_z);
    keys[i] = key;
    pids[i] = i;
</%def>

#################################################################################

<%def name="find_num_unique_cids_args(data_t)" cached="True">
    unsigned long* keys
</%def>

<%def name="map_find_num_unique_cids_src(data_t)" cached="True">
    i != 0 && keys[i] != keys[i - 1] ? 1 : 0
</%def>

<%def name="fill_unique_cids_args(data_t)" cached="True">
    unsigned long* keys, unsigned int* cids, unsigned int* start_cid
</%def>

<%def name="inp_fill_unique_cids_src(data_t)" cached="True">
    i != 0 && keys[i] != keys[i - 1] ? 1 : 0
</%def>

<%def name="out_fill_unique_cids_src(data_t)" cached="True">
    cids[i] = item;
    if(item != prev_item)
        start_cid[item] = i;
</%def>


<%def name="fill_length_cids_args(data_t)" cached="True">
    unsigned int* start_cids, unsigned int* lengths,
    unsigned int num_cids, unsigned int num_particles
</%def>

<%def name="fill_length_cids_src(data_t)" cached="True">
    lengths[i] = (i < num_cids - 1 ? start_cids[i + 1] : num_particles) - start_cids[i] 
</%def>

#################################################################################

<%def name="map_cid_to_idx_args(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, int num_particles,
    ${data_t} cell_size, ${data_t}3 min, unsigned int* pids,
    unsigned long* keys, unsigned int* cids, int* cid_to_idx
</%def>

<%def name="map_cid_to_idx_src(data_t)" cached="True">
    unsigned int cid = cids[i];

    if(i != 0 && cids[i - 1] == cid)
        PYOPENCL_ELWISE_CONTINUE;

    unsigned int j;
    int idx;
    unsigned long key;
    int nbr_boxes_length;
    ${data_t}3 c;

    unsigned int pid = pids[i];

    FIND_CELL_ID(
        x[pid] - min.x,
        y[pid] - min.y,
        z[pid] - min.z,
        cell_size, c.x, c.y, c.z
        );

    unsigned long* nbr_boxes[27];

    nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes);

    #pragma unroll
    for(j=0; j<nbr_boxes_length; j++)
    {
        key = nbr_boxes[j];
        idx = find_idx(keys, num_particles, key);
        cid_to_idx[27*cid + j] = idx;
    }
</%def>

<%def name="map_dst_to_src_args(data_t)" cached="True">
    unsigned int* dst_to_src, unsigned int* cids_dst, int* cid_to_idx_dst,
    unsigned long* keys_dst, unsigned long* keys_src, unsigned int* cids_src,
    unsigned int num_particles_src, int* max_cid_src
</%def>

<%def name="map_dst_to_src_src(data_t)" cached="True">
    int idx = cid_to_idx_dst[27*i];
    unsigned long key = keys_dst[idx];
    int idx_src = find_idx(keys_src, num_particles_src, key);
    dst_to_src[i] = (idx_src == -1) ? atomic_inc(&max_cid_src[0]) : cids_src[idx_src];
</%def>

<%def name="fill_overflow_map_args(data_t)" cached="True">
    unsigned int* dst_to_src, int* cid_to_idx_dst, ${data_t}* x,
    ${data_t}* y, ${data_t}* z, int num_particles_src,
    ${data_t} cell_size, ${data_t}3 min, unsigned long* keys_src,
    unsigned int* pids_dst, int* overflow_cid_to_idx,
    unsigned int max_cid_src
</%def>

<%def name="fill_overflow_map_src(data_t)" cached="True">
    unsigned int cid = dst_to_src[i];
    // i is the cid in dst

    if(cid < max_cid_src)
        PYOPENCL_ELWISE_CONTINUE;

    int idx = cid_to_idx_dst[27*i];

    unsigned int j;
    unsigned long key;
    int nbr_boxes_length;
    ${data_t}3 c;

    unsigned int pid = pids_dst[idx];

    FIND_CELL_ID(
        x[pid] - min.x,
        y[pid] - min.y,
        z[pid] - min.z,
        cell_size, c.x, c.y, c.z
        );

    unsigned long* nbr_boxes[27];

    nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes);

    unsigned int start_idx = cid - max_cid_src;

    #pragma unroll
    for(j=0; j<nbr_boxes_length; j++)
    {
        key = nbr_boxes[j];
        idx = find_idx(keys_src, num_particles_src, key);
        overflow_cid_to_idx[27*start_idx + j] = idx;
    }
</%def>

<%def name="z_order_nbrs_prep(data_t, sorted, dst_src, wgs)", cached="False">
    unsigned int cid = (unsigned int) get_group_id(0);

    if(cid >= max_cid_dst)
        return;

    long np_wg = (long) lengths_dst[cid];

    long start_wg = (long) cid_to_idx_dst[27 * cid];

    unsigned int qid = 0;
    ${data_t}4 q;
    ${data_t} h_i;
    ${data_t} h_j;

    // Only the threads with lid < np_wg are responsible for
    // finding neighbors. Other threads are only used for
    // copying data to local memory
    if(lid < np_wg)
    {
        qid = pids_dst[start_wg + lid];
        q = (${data_t}4)(d_x[qid], d_y[qid], d_z[qid], d_h[qid]);
        h_i = radius_scale2*q.w*q.w;
    }

    int3 c;

    __local ${data_t} xlocal[${wgs}];
    __local ${data_t} ylocal[${wgs}];
    __local ${data_t} zlocal[${wgs}];
    __local ${data_t} hlocal[${wgs}];

    int idx;
    unsigned int j;
    ${data_t} dist;

    unsigned long key;
    unsigned int pid;
    unsigned int curr_length;
    unsigned int curr_cid, k;

    __global int* nbr_boxes = cid_to_idx;
    unsigned int start_id_nbr_boxes;

    % if dst_src:
        cid = dst_to_src[cid];
        start_id_nbr_boxes = 27*cid;
        if(cid >= max_cid_src)
        {
            start_id_nbr_boxes = 27*(cid - max_cid_src);
            nbr_boxes = overflow_cid_to_idx;
        }
    % else:
        start_id_nbr_boxes = 27*cid;
    % endif

</%def>

<%def name="z_order_nbr_lengths_args(data_t)" cached="False">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, unsigned long* keys,
    unsigned int* pids_dst, unsigned int* pids_src, unsigned int max_cid_src,
    unsigned int max_cid_dst, int* cid_to_idx_dst,
    unsigned int* cids, int* cid_to_idx, int* overflow_cid_to_idx,
    unsigned int* dst_to_src, unsigned int* lengths_dst,
    unsigned int* lengths_src,
    unsigned int* nbr_lengths, ${data_t} radius_scale2,
    ${data_t} cell_size, unsigned int* working_qids
</%def>

<%def name="z_order_nbr_lengths_src(data_t, sorted, dst_src, wgs)" cached="False">
    ${z_order_nbrs_prep(data_t, sorted, dst_src, wgs)}

    if(i < num_particles)
        working_qids[i] = (unsigned int) i;

    unsigned int nbr_length = 0;

    ${data_t}4 s;

    int bid, nblocks;
    unsigned int m, pid_idx;

    for(j=0; j<27; j++)
    {
        idx = nbr_boxes[start_id_nbr_boxes + j];
        if(idx == -1)
            continue;
        key = keys[idx];
        curr_cid = cids[idx];

        curr_length = lengths_src[curr_cid];
        nblocks = ceil(((float) curr_length) / ${wgs});

        // In the current implementation, nblocks will always
        // be equal to 1 as wgs is the max number of particles
        // per cell. This may not always work as there can be
        // a large number of particles per cell. In the future,
        // work items in such cells can be made to handle multiple
        // particles. In that case nblocks may not always be 1
        for(bid=0; bid<nblocks; bid++)
        {
            pid_idx = (unsigned int) (bid * ${wgs} + lid);
            if(pid_idx >= curr_length)
                break;
            pid = pids_src[idx + pid_idx];
            xlocal[lid] = s_x[pid];
            ylocal[lid] = s_y[pid];
            zlocal[lid] = s_z[pid];
            hlocal[lid] = s_h[pid];

            barrier(CLK_LOCAL_MEM_FENCE);

            m = curr_length < ${wgs} ? curr_length : ${wgs};
                
            if(lid < np_wg)
            {
                for(k=0; k<m; k++)
                {
                    pid = pids_src[idx + k];
                    s = (${data_t}4)(xlocal[k], ylocal[k],
                                     zlocal[k], hlocal[k]);
                    h_j = radius_scale2 * s.w * s.w;
                    dist = NORM2(q.x - s.x, q.y - s.y, q.z - s.z);
                    if(dist < h_i || dist < h_j)
                        nbr_length++;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

    if(lid < np_wg)
        nbr_lengths[qid] = 0;

</%def>


<%def name="z_order_nbrs_args(data_t)" cached="False">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, unsigned long* keys,
    unsigned int* pids_dst, unsigned int* pids_src, unsigned int max_cid_src,
    unsigned int max_cid_dst, int* cid_to_idx_dst,
    unsigned int* cids, int* cid_to_idx, int* overflow_cid_to_idx,
    unsigned int* dst_to_src, unsigned int* lengths_dst,
    unsigned int* lengths_src,
    unsigned int* start_indices, unsigned int* nbrs,
    ${data_t} radius_scale2, ${data_t} cell_size
</%def>

<%def name="z_order_nbrs_src(data_t, sorted, dst_src)" cached="False">
    ${z_order_nbrs_prep(data_t, sorted, dst_src, wgs)}

    unsigned long start_idx = (unsigned long) start_indices[qid];
    unsigned long curr_idx = 0;

    ${data_t}4 s;

    int m, pid_idx, bid, nblocks;

    for(j=0; j<27; j++)
    {
        idx = nbr_boxes[start_id_nbr_boxes + j];
        if(idx == -1)
            continue;
        key = keys[idx];
        curr_cid = cids[idx];

        curr_length = lengths_src[curr_cid];
        nblocks = ceil(((float) curr_length) / ${wgs});

        // In the current implementation, nblocks will always
        // be equal to 1 as wgs is the max number of particles
        // per cell. This may not always work as there can be
        // a large number of particles per cell. In the future,
        // work items in such cells can be made to handle multiple
        // particles. In that case nblocks may not always be 1
        for(bid=0; bid<nblocks; bid++)
        {
            pid_idx = bid * ${wgs} + lid;
            if(pid_idx >= curr_length)
                break;
            pid = pids_src[idx + pid_idx];
            xlocal[lid] = s_x[pid];
            ylocal[lid] = s_y[pid];
            zlocal[lid] = s_z[pid];
            hlocal[lid] = s_h[pid];

            barrier(CLK_LOCAL_MEM_FENCE);

            m = curr_length < ${wgs} ? curr_length : ${wgs};
                
            if(lid < np_wg)
            {
                for(k=0; k<m; k++)
                {
                    pid = pids_src[idx + k];
                    s = (${data_t}4)(xlocal[k], ylocal[k],
                                     zlocal[k], hlocal[k]);
                    h_j = radius_scale2 * s.w * s.w;
                    dist = NORM2(q.x - s.x, q.y - s.y, q.z - s.z);
                    if(dist < h_i || dist < h_j)
                    {
                        nbrs[start_idx + curr_idx] = pid;
                        curr_idx++;
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

</%def>

