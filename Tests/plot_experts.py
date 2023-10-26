import matplotlib.pyplot as plt
import torch
import numpy as np
import os
# Data

experts = np.arange(16)
experts_hit = np.zeros(16)
iter = 30
start = 30
end_plus_one = 31
iteration_gap = 15
for iter in range(start,end_plus_one,iteration_gap):

    for rank_id in range(1):
        print(f'---------- rank_id = {rank_id}, iter = {iter}----------------')
        # dispatch_mask = torch.load('/Users/mark/Work/Research/OOC/local_analysis/dispatch_mask_rank'+str(rank_id)+'.pth', map_location=torch.device('cpu'))
        dispatch_mask = torch.load('/shared_ssd_storage/shilonglei/OOC/Jobs/output/dispatch_mask_1/dispatch_mask_iter'+str(iter)+'/dispatch_mask_layer4_rank0_iter'+str(iter)+'.pth', map_location=torch.device('cpu'))
        print(dispatch_mask.shape)
        # [token_num, expert_num, capacity]
        for token_id in range(dispatch_mask.shape[0]):
            if token_id % 10 == 0: print(f"analyzing token_id = {token_id}")
            for expert_id in range(dispatch_mask.shape[1]):
                found_flag = False
                for capacity_id in range(dispatch_mask.shape[2]):
                    if dispatch_mask[token_id, expert_id, capacity_id] == True:
                        experts_hit[expert_id] = experts_hit[expert_id] + 1
                        found_flag = True
                        break
                if found_flag == True:
                    break

    # Create a histogram
    plt.bar(experts, experts_hit, color='skyblue')

    # Add labels and title
    plt.xlabel('Expert id')
    plt.ylabel('Hit')
    plt.title('Expert Hit Histogram')

    save_fig_root_path = '/shared_ssd_storage/shilonglei/OOC/Jobs/output/dispatch_mask_1/Figures'
    if not os.path.exists(save_fig_root_path):
        os.mkdir(save_fig_root_path)
    plt.savefig(save_fig_root_path + '/dispatch_mask_layer4_rank0_iter'+str(iter)+'.png')
    # Show the plot
    # plt.show()