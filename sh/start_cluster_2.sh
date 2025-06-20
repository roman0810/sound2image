if tmux has-session -t EDITOR 2>/dev/null; then
    # Если существует — убиваем её
    tmux kill-session -t EDITOR
fi


tmux has-session -t development

if [ $? != 0  ]; then
    tmux new-session -s EDITOR -n Desktop -d

    # Set up main EDITOR window
    tmux select-window -t EDITOR:Desktop


    # Create splits (must executed outside of the session)
    tmux new-window -t EDITOR:Desktop
    tmux split-window -v -t EDITOR
    tmux split-window -v -t EDITOR
    tmux split-window -v -t EDITOR

    tmux select-layout -t EDITOR:Desktop even-vertical


    # Подготовка master-node
    tmux select-pane -t 0
    tmux send-keys -t EDITOR:Desktop.0 'ssh usr5@10.162.1.91' C-m
    tmux send-keys -t EDITOR:Desktop.0 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t EDITOR:Desktop.0 'source ~/myenv/bin/activate' C-m
    tmux send-keys -t EDITOR:Desktop.0 'export NCCL_SOCKET_IFNAME=eno1' C-m

    # Первое SSH-подключение и выполнение команд
    tmux select-pane -t 1

    tmux send-keys -t EDITOR:Desktop.1 'ssh usr6@10.162.1.92' C-m
    tmux send-keys -t EDITOR:Desktop.1 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t EDITOR:Desktop.1 'source ~/myenv/bin/activate' C-m
    tmux send-keys -t EDITOR:Desktop.1 'export NCCL_SOCKET_IFNAME=eno1' C-m

    # Второе SSH-подключение и выполнение команд
    tmux select-pane -t 2

    tmux send-keys -t EDITOR:Desktop.2 'ssh usr7@10.162.1.93' C-m
    tmux send-keys -t EDITOR:Desktop.2 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t EDITOR:Desktop.2 'source ~/myenv/bin/activate' C-m
    tmux send-keys -t EDITOR:Desktop.2 'export NCCL_SOCKET_IFNAME=eno1' C-m

    # Третье SSH-подключение и выполнение команд
    tmux select-pane -t 3

    tmux send-keys -t EDITOR:Desktop.3 'ssh usr8@10.162.1.94' C-m
    tmux send-keys -t EDITOR:Desktop.3 'cd ~/Documents/GitHub/sound2image' C-m
    tmux send-keys -t EDITOR:Desktop.3 'source ~/myenv/bin/activate' C-m
    tmux send-keys -t EDITOR:Desktop.3 'export NCCL_SOCKET_IFNAME=eno1' C-m

fi
# Присоединяемся к сессии
tmux attach -t EDITOR
