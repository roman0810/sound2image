if tmux has-session -t CP_MONITOR 2>/dev/null; then
    # Если существует — убиваем её
    tmux kill-session -t CP_MONITOR
fi


tmux has-session -t development

if [ $? != 0  ]; then
    tmux new-session -s CP_MONITOR -n Desktop -d

    # Set up main CP_MONITOR window
    tmux select-window -t CP_MONITOR:Desktop


    # Create splits (must executed outside of the session)
    tmux new-window -t CP_MONITOR:Desktop
    tmux split-window -v -t CP_MONITOR
    tmux split-window -v -t CP_MONITOR
    tmux split-window -v -t CP_MONITOR

    tmux select-layout -t CP_MONITOR:Desktop even-vertical


    # Подготовка master-node
    tmux select-pane -t 0
    tmux send-keys -t CP_MONITOR:Desktop.0 'ssh usr5@10.162.1.91' C-m
    tmux send-keys -t CP_MONITOR:Desktop.0 'htop' C-m

    # Первое SSH-подключение и выполнение команд
    tmux select-pane -t 1

    tmux send-keys -t CP_MONITOR:Desktop.1 'ssh usr6@10.162.1.92' C-m
    tmux send-keys -t CP_MONITOR:Desktop.1 'htop' C-m

    # Второе SSH-подключение и выполнение команд
    tmux select-pane -t 2

    tmux send-keys -t CP_MONITOR:Desktop.2 'ssh usr7@10.162.1.93' C-m
    tmux send-keys -t CP_MONITOR:Desktop.2 'htop' C-m

    # Третье SSH-подключение и выполнение команд
    tmux select-pane -t 3

    tmux send-keys -t CP_MONITOR:Desktop.3 'ssh usr8@10.162.1.94' C-m
    tmux send-keys -t CP_MONITOR:Desktop.3 'htop' C-m

fi
# Присоединяемся к сессии
tmux attach -t CP_MONITOR
