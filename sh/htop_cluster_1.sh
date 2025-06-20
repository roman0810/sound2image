#!/bin/sh

# Note: the default window, pane index start from 0
# you may need to modify the following index if you changed it in .tmux.conf
# E.g.
# set -g base-index 1 # start window index at 1
# setw -g pane-base-index 1 # pane index starts at 1


# Проверяем, существует ли сессия с именем "CP_Monitor"
if tmux has-session -t CP_Monitor 2>/dev/null; then
    # Если существует — убиваем её
    tmux kill-session -t CP_Monitor
fi


tmux has-session -t development

if [ $? != 0  ]; then
    tmux new-session -s CP_Monitor -n Desktop -d

    # Set up main CP_Monitor window
    tmux select-window -t CP_Monitor:Desktop


    # Create splits (must executed outside of the session)
    tmux new-window -t CP_Monitor:Desktop
    tmux split-window -v -t CP_Monitor
    tmux split-window -v -t CP_Monitor
    tmux split-window -v -t CP_Monitor

    tmux select-layout -t CP_Monitor:Desktop even-vertical


    # Подготовка master-node
    tmux select-pane -t 0
    tmux send-keys -t CP_Monitor:Desktop.0 'htop' C-m

    # Первое SSH-подключение и выполнение команд
    tmux select-pane -t 1

    tmux send-keys -t CP_Monitor:Desktop.1 'ssh usr2@10.162.1.82' C-m
    tmux send-keys -t CP_Monitor:Desktop.1 'htop' C-m

    # Второе SSH-подключение и выполнение команд
    tmux select-pane -t 2

    tmux send-keys -t CP_Monitor:Desktop.2 'ssh usr3@10.162.1.71' C-m
    tmux send-keys -t CP_Monitor:Desktop.2 'htop' C-m

    # Третье SSH-подключение и выполнение команд
    tmux select-pane -t 3

    tmux send-keys -t CP_Monitor:Desktop.3 'ssh usr4@10.162.1.51' C-m
    tmux send-keys -t CP_Monitor:Desktop.3 'htop' C-m

fi
# Присоединяемся к сессии
tmux attach -t CP_Monitor

