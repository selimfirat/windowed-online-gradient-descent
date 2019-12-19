python main.py --mlflow --experiment_name all_datasets --data pumadyn --truncation 30 --hidden_size 16 --optimizer adam --lr_ih 0.004 --lr_hh 0.004 --lr_oh 0.004 --num_threads -1 &
python main.py --mlflow --experiment_name all_datasets --data pumadyn --truncation 30 --hidden_size 16 --optimizer rmsprop --lr_ih 0.004 --lr_hh 0.004 --lr_oh 0.004 --num_threads -1 &
python main.py --mlflow --experiment_name all_datasets --data pumadyn --truncation 30 --hidden_size 16 --optimizer sgd --lr_ih 0.03 --lr_hh 0.03 --lr_oh 0.03 --num_threads -1 &
python main.py --mlflow --experiment_name all_datasets --data pumadyn --truncation 30 --hidden_size 16 --optimizer sgd --lr_ih 0.05 --lr_hh 0.05 --lr_oh 0.05 --num_threads -1 &


python main.py --data elevator --lr_hh 0.5 --lr_oh 1.0 --lr_ih 0.5 --truncation 150 --alpha 0 --window_size 120 --optimizer windowed_ogd --hidden_size 16 --time_decay --append_input
python main.py --data elevator --lr_hh 0.5 --lr_oh 1.0 --lr_ih 0.5 --truncation 150 --alpha 0 --window_size 120 --optimizer windowed_ogd --hidden_size 16 --time_decay



# elevator
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/75d0ff0febf0460c87d2b2471ced7603/artifacts/75d0ff0febf0460c87d2b2471ced7603.csv .
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/eb0bbdc3a618431d8188cb3d8fd67a0e/artifacts/eb0bbdc3a618431d8188cb3d8fd67a0e.csv .
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/3e0bd4dcdd644000aee2dd99007e7cc7/artifacts/3e0bd4dcdd644000aee2dd99007e7cc7.csv .
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/9ad7760b28634d2bb6fefe46528f915a/artifacts/9ad7760b28634d2bb6fefe46528f915a.csv .


# puma32f
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/980cd4778c18470d9bc4479cb93d969f/artifacts/980cd4778c18470d9bc4479cb93d969f.csv .
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/682305a65b0a43aeae90a38a5c0602ee/artifacts/682305a65b0a43aeae90a38a5c0602ee.csv .
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/04af63987d6d4b1dbb127774ca55b5b8/artifacts/04af63987d6d4b1dbb127774ca55b5b8.csv .
scp selim@192.168.2.210:/home/selim/tmp/pycharm_project_372/mlruns/19/ed407b0831ca4f7197d2a369c42cc36e/artifacts/ed407b0831ca4f7197d2a369c42cc36e.csv .
