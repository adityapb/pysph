//CL//

<%def name="get_helpers(data_t, if_cuda)" cached="True">
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))

    #define FIND_CELL_ID(x, y, z, h, c_x, c_y, c_z) \
        c_x = floor((x)/h); c_y = floor((y)/h); c_z = floor((z)/h)

    %if if_cuda:
        #define ATOMIC_INC(x) atomicAdd(x, 1)
        #define ELWISE_CONTINUE continue
        #define MAKE_VEC(v1, v2, v3, v4) make_${data_t}4(v1, v2, v3, v4)
    %else:
        #define ATOMIC_INC(x) atomic_inc(x)
        #define ELWISE_CONTINUE PYOPENCL_ELWISE_CONTINUE
        #define MAKE_VEC(v1, v2, v3, v4) (${data_t}4)(v1, v2, v3, v4)
    %endif


    WITHIN_KERNEL unsigned long interleave(unsigned long p, \
            unsigned long q, unsigned long r);

    WITHIN_KERNEL int neighbor_boxes(int c_x, int c_y, int c_z, \
            unsigned long* nbr_boxes);

    WITHIN_KERNEL unsigned long interleave(unsigned long p, \
            unsigned long q, unsigned long r)
    {
        p = (p | (p << 32)) & 0x1f00000000ffff;
        p = (p | (p << 16)) & 0x1f0000ff0000ff;
        p = (p | (p <<  8)) & 0x100f00f00f00f00f;
        p = (p | (p <<  4)) & 0x10c30c30c30c30c3;
        p = (p | (p <<  2)) & 0x1249249249249249;

        q = (q | (q << 32)) & 0x1f00000000ffff;
        q = (q | (q << 16)) & 0x1f0000ff0000ff;
        q = (q | (q <<  8)) & 0x100f00f00f00f00f;
        q = (q | (q <<  4)) & 0x10c30c30c30c30c3;
        q = (q | (q <<  2)) & 0x1249249249249249;

        r = (r | (r << 32)) & 0x1f00000000ffff;
        r = (r | (r << 16)) & 0x1f0000ff0000ff;
        r = (r | (r <<  8)) & 0x100f00f00f00f00f;
        r = (r | (r <<  4)) & 0x10c30c30c30c30c3;
        r = (r | (r <<  2)) & 0x1249249249249249;

        return (p | (q << 1) | (r << 2));
    }

    WITHIN_KERNEL int find_idx(GLOBAL_MEM unsigned long* keys, \
            int num_particles, unsigned long key)
    {
        int first = 0;
        int last = num_particles - 1;
        int middle = (first + last) / 2;

        while(first <= last)
        {
            if(keys[middle] < key)
                first = middle + 1;
            else if(keys[middle] > key)
                last = middle - 1;
            else if(keys[middle] == key)
            {
                if(middle == 0)
                    return 0;
                if(keys[middle - 1] != key)
                    return middle;
                else
                    last = middle - 1;
            }
            middle = (first + last) / 2;
        }

        return -1;
    }

    WITHIN_KERNEL int neighbor_boxes(int c_x, int c_y, int c_z, \
        unsigned long* nbr_boxes)
    {
        int nbr_boxes_length = 1;
        int j, k, m;
        unsigned long key;
        nbr_boxes[0] = interleave(c_x, c_y, c_z);

        #pragma unroll
        for(j=-1; j<2; j++)
        {
            #pragma unroll
            for(k=-1; k<2; k++)
            {
                #pragma unroll
                for(m=-1; m<2; m++)
                {
                    if((j != 0 || k != 0 || m != 0) && c_x+m >= 0 && c_y+k >= 0 && c_z+j >= 0)
                    {
                        key = interleave(c_x+m, c_y+k, c_z+j);
                        nbr_boxes[nbr_boxes_length] = key;
                        nbr_boxes_length++;
                    }
                }
            }
        }

        return nbr_boxes_length;
    }

</%def>


