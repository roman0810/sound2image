#!/bin/sh

# Note: the default window, pane index start from 0
# you may need to modify the following index if you changed it in .tmux.conf
# E.g.
# set -g base-index 1 # start window index at 1
# setw -g pane-base-index 1 # pane index starts at 1


# Проверяем, существует ли сессия с именем "NV_Monitor"
if tmux has-session -t NV_Monitor 2>/dev/null; then
    # Если существует — убиваем её
    tmux kill-session -t NV_Monitor
fi


tmux has-session -t development

if [ $? != 0  ]; then
    tmux new-session -s NV_Monitor -n Desktop -d

    # Set up main NV_Monitor window
    tmux select-window -t NV_Monitor:Desktop


    # Create splits (must executed outside of the session)
    tmux new-window -t NV_Monitor:Desktop
    tmux split-window -v -t NV_Monitor
    tmux split-window -v -t NV_Monitor
    tmux split-window -v -t NV_Monitor

    tmux select-layout -t NV_Monitor:Desktop even-vertical


    # Подготовка master-node
    tmux select-pane -t 0
    tmux send-keys -t NV_Monitor:Desktop.0 'nvtop' C-m

    # Первое SSH-подключение и выполнение команд
    tmux select-pane -t 1

    tmux send-keys -t NV_Monitor:Desktop.1 'ssh usr2@10.162.1.82' C-m
    tmux send-keys -t NV_Monitor:Desktop.1 'nvtop' C-m

    # Второе SSH-подключение и выполнение команд
    tmux select-pane -t 2

    tmux send-keys -t NV_Monitor:Desktop.2 'ssh usr3@10.162.1.71' C-m
    tmux send-keys -t NV_Monitor:Desktop.2 'nvtop' C-m

    # Третье SSH-подключение и выполнение команд
    tmux select-pane -t 3

    tmux send-keys -t NV_Monitor:Desktop.3 'ssh usr4@10.162.1.51' C-m
    tmux send-keys -t NV_Monitor:Desktop.3 'nvtop' C-m

fi
# Присоединяемся к сессии
tmux attach -t NV_Monitor

