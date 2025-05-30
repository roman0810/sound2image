#!/bin/sh

# Note: the default window, pane index start from 0
# you may need to modify the following index if you changed it in .tmux.conf
# E.g.
# set -g base-index 1 # start window index at 1
# setw -g pane-base-index 1 # pane index starts at 1


# Проверяем, существует ли сессия с именем "Editor"
if tmux has-session -t Editor 2>/dev/null; then
    # Если существует — убиваем её
    tmux kill-session -t Editor
fi


tmux has-session -t development

if [ $? != 0  ]; then
    tmux new-session -s Editor -n Desktop -d

    # Set up main editor window
    tmux select-window -t Editor:Desktop


    # Create splits (must executed outside of the session)
    tmux new-window -t Editor:Desktop
    tmux split-window -v -t Editor
    tmux split-window -v -t Editor
    tmux split-window -v -t Editor

    tmux select-layout -t Editor:Desktop even-vertical


    # Подготовка master-node
    tmux select-pane -t 0
    tmux send-keys -t Editor:Desktop.0 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t Editor:Desktop.0 'export NCCL_SOCKET_IFNAME=eno1' C-m

    # Первое SSH-подключение и выполнение команд
    tmux select-pane -t 1

    tmux send-keys -t Editor:Desktop.1 'ssh usr2@10.162.1.82' C-m
    tmux send-keys -t Editor:Desktop.1 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t Editor:Desktop.1 'export NCCL_SOCKET_IFNAME=eno1' C-m

    # Второе SSH-подключение и выполнение команд
    tmux select-pane -t 2

    tmux send-keys -t Editor:Desktop.2 'ssh usr3@10.162.1.71' C-m
    tmux send-keys -t Editor:Desktop.2 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t Editor:Desktop.2 'source ~/myenv/bin/activate' C-m
    tmux send-keys -t Editor:Desktop.2 'export NCCL_SOCKET_IFNAME=eno1' C-m

    # Третье SSH-подключение и выполнение команд
    tmux select-pane -t 3

    tmux send-keys -t Editor:Desktop.3 'ssh usr4@10.162.1.51' C-m
    tmux send-keys -t Editor:Desktop.3 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t Editor:Desktop.3 'conda deactivate' C-m
    tmux send-keys -t Editor:Desktop.3 'source ~/myenv/bin/activate' C-m
    tmux send-keys -t Editor:Desktop.3 'export NCCL_SOCKET_IFNAME=eno1' C-m

fi
# Присоединяемся к сессии
tmux attach -t Editor
