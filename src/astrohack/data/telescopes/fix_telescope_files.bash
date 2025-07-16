#!/usr/bin/bash

f_to_corr=(".zattrs" ".zmetadata")

old_vars=("secondary_dist")
new_vars=("secondary_distance_to_focus")

# old_vars=("diam" "ant_list" "filename" "filepath" "inlim" "oulim"
# "dist_dict" "inrad" "ourad" "npanel" "nrings")

# new_vars=("diameter" "antenna_list" "file_name" "file_path"
# "inner_radial_limit" "outer_radial_limit" "station_distance_dict"
# "panel_inner_radii" "panel_outer_radii" "n_panel_per_ring" "n_rings_of_panels")

nvars=$((${#old_vars[*]}-1))
echo $nvars

for tele in *.zarr; do
    pushd ${tele} > /dev/null

    echo "Fixing ${tele}..."
    pwd

    for filename in ${f_to_corr[@]}; do

	for i in $(seq 0 ${nvars}); do
	    #grep ${old_vars[i]} $filename
	    cmd="sed -i 's/${old_vars[i]}/${new_vars[i]}/g' $filename"
	    eval ${cmd}
	done
    done
    
    
    popd  > /dev/null
done

