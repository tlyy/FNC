for i in 1 2 3 4 5; do
    for game in Freeway Gopher Kangaroo KungFuMaster; do
        python -m bbf.train \
                --agent=FNC \
                --gin_files=bbf/configs/FNC.gin \
                --base_dir=exp/fnc/$game/$i \
                --gin_bindings="DataEfficientAtariRunner.game_name = '$game'"
    done
done



# export MUJOCO_GL=osmesa
# for i in 1 2 3 4 5; do
#     for game in cheetah-run cartpole-swingup reacher-easy finger-spin ball_in_cup-catch walker-walk; do
#         python -m continuous_control.train \
#                 --save_dir=exp_con/fnc/$game/$i \
#                 --env_name $game \
#                 --max_steps 100000 \
#                 --resets \
#                 --reset_interval 20000 \
#                 --fnc \
#                 --threshold 0.0 \
#                 --dnr_weight 0.8
#     done
# done