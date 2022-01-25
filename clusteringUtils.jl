_sq_dist(x, y) = (x - y)^2

function dtwCTM(a, b, window; dist_fn = _sq_dist)
    # Calculate cost matrix
    n, m = length(a), length(b)
    
    # Create the sakoechiba band
    window = abs(n-m) > window ? abs(n-m) : window
    
    region1 = [max(1, i-window) for i in 1:n]
    region2 = [min(m, i+window) for i in 1:n]
    
    # Initialize cost matrix
    cost_mat_ = ones(n, m) .* Inf

    for i in 1:n
        for j in region1[i]:region2[i]
            cost_mat_[i, j] = dist_fn(a[i], b[i])
        end
    end
    
    # Calculate accumulated cost matrix
    acc_cost_mat = ones(n, m) .* Inf
    
    acc_cost_mat[1, 1:region2[1]] = cumsum(cost_mat_[1, 1:region2[1]])
    acc_cost_mat[1:region2[1], 1] = cumsum(cost_mat_[1:region2[1], 1])
    
    region1_ = [max(2, i-window) for i in 1:n]
    
    for i in 2:n
        for j in region1_[i]:region2[i]
            # SymmetricP0 step pattern recursion
            acc_cost_mat[i, j] =  min(acc_cost_mat[i-1, j-1] + 2*cost_mat_[i, j],
                                      acc_cost_mat[i, j-1] + cost_mat_[i, j], 
                                      acc_cost_mat[i-1, j] + cost_mat_[i, j])            
        end
    end
    
    return acc_cost_mat[end, end]
end

function _drawDendLines(ax, pos1, pos2, heights, stage)
    using CairoMakie
    line_lefts_ = []

    ## Draw horizontal lines ##
    
    # Line 1
    right_x, right_y = pos1
    left_x, left_y = heights[stage], right_y

    lines!(ax, [left_x, right_x], [left_y, right_y], color=:black)
    push!(line_lefts_, (left_x, left_y))

    # Line 2
    right_x, right_y = pos2
    left_x, left_y = heights[stage], right_y
    lines!(ax, [left_x, right_x], [left_y, right_y], color=:black)
    push!(line_lefts_, (left_x, left_y))

    ## Draw vertical line ##
    vline_top_x, vline_top_y = line_lefts_[1]
    _, vline_bot_y = line_lefts_[2]
    lines!(ax, [vline_top_x, vline_top_x], [vline_top_y, vline_bot_y], color=:black)
    
    # reutrn the node location for later use
    return (vline_top_x, (vline_bot_y + vline_top_y)/2)
end

function drawDendrogram(hclust::Hclust, ax::Axis; feature_names=nothing)
    using CairoMakie
    using Clustering

    heights_   = hclust.height
    merge_mat  = hclust.merge
    order_     = hclust.order
    num_stages = size(merge_mat)[1]
    
    if feature_names === nothing
        labels = "Feature " .* string.(order_)
    else
        labels = feature_names[order_]
    end
    
    # Draw leaves
    leaf_locations = Dict()
    xpos_start = 0
    for marker_pos in 1:length(order_)
        leaf_locations[order_[marker_pos]] = (0, xpos_start)
        text!(ax, labels[marker_pos], position = (-1, xpos_start), color=:black,
                align=(:right, :center), font="Arial", textsize=15, offset=(-1, 0))

        xpos_start += 10
    end
    
    # Join the leaves/nodes hierarchically using lines
    node_locations = Dict()
    
    for stage in 1:num_stages
        row_ = merge_mat[stage, :]
        
        if all(row_ .< 0) # behavior 1            
            pos1, pos2 = leaf_locations[-row_[1]], leaf_locations[-row_[2]]
        elseif all(row_ .> 0) #behavior 2            
            pos1, pos2 = node_locations[row_[1]], node_locations[row_[2]]            
        else #behavior 3            
            pos1, pos2 = leaf_locations[-row_[1]], node_locations[row_[2]]
        end
        
        node_locations[stage] = _drawDendLines(ax, pos1, pos2, heights_, stage)
    end
    
    return node_locations
end