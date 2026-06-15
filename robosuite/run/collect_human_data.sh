python /home/dodo/Documents/DAggar/robosuite/robosuite/scripts/collect_human_demonstrations.py \
    --directory /home/dodo/Documents/DAggar/robosuite/data2/PickPlaceMilk/expert \
    --robots Panda \
    --environment PickPlaceMilk \
    --device spacemouse 

# GR1 views:
# 'frontview', 'birdview', 'agentview', 'robot0_obs_hands', 'robot0_overshoulder', 'robot0_behindhead', 
# 'robot0_robotview', 'robot0_eye_in_right_hand', 'robot0_eye_in_left_hand'
# python /home/dodo/Documents/DAggar/robosuite/robosuite/scripts/collect_human_demonstrations.py \
#     --directory /home/dodo/Documents/DAggar/robosuite/data/PickPlaceCereal\
#     --robots GR1 \
#     --environment Lift \
#     --device spacemouse \
#     --camera robot0_robotview \
#     --arm right \
#     --controller BASIC
