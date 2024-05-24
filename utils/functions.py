import matplotlib.pyplot as plt
import numpy as np

def get_ball_and_paddle_coordinates(state, print_state=False):
    # Extract ball and paddle coordinates from the game state
    ball_x = int(state[49])
    ball_y = int(state[54])
    cpu_paddle_y = int(state[21])
    player_paddle_y = int(state[51])
    
    if print_state:
        print(f"Ball position: ({ball_x}, {ball_y})")
        print(f"CPU paddle Y position: {cpu_paddle_y}")
        print(f"Player paddle Y position: {player_paddle_y}")

    return np.array([ball_x, ball_y, cpu_paddle_y, player_paddle_y])

def act2cat(act):
    # Map action values to categories
    dicmap = {0 : 0, 1 : 0, 2 : 1, 3 : 2, 4 : 1, 5 : 2}
    return dicmap[act]

def cat2act(cat):
    # Map categories to action values
    dicmap = {0 : 0, 1 : 2, 2 : 3}
    return dicmap[cat]

def plot_policy(OUT):
    plt.figure(figsize=(18,18))
    plt.plot(np.array(OUT))
    plt.ylim(-.1,1.1)
    plt.xlabel('time')
    plt.ylabel('policy')

    plt.savefig('policy.png', facecolor='w', edgecolor='w')
    plt.close()
    return

def plot_spikes(S):
    plt.figure(figsize=(18,18))
    
    plt.subplot(121)
    plt.imshow(1-np.array(S)[:,0:512].T,aspect='auto',cmap ='gray')
    
    num_events = np.sum(S, axis=0)
    plt.subplot(122)
    plt.barh(np.arange(512), num_events)
    plt.xlabel("num events")
    plt.ylabel("neuron id")
    plt.savefig('spikes.png', facecolor='w', edgecolor='w')
    plt.close()
    return

def plot_rewards(REWARDS,REWARDS_MEAN,S_agent,S_model,OUT,RAM,RAM_PRED,R,R_PRED,ENTROPY_MEAN,filename = 'figure.png'):

    plt.figure(figsize=(8, 11), dpi=100)
    plt.suptitle('Number of spikes: ' + str(np.sum(S_agent)), fontsize=11)

    plt.subplot(4,2,1)
    # create a list from 0 to 2000 in steps of 50
    x = np.linspace(50, 50*len(REWARDS_MEAN), len(REWARDS_MEAN))
    plt.plot(x, REWARDS_MEAN)
    plt.ylabel('reward mean')
    # plt.xlabel('iterations')

    plt.subplot(4,2,2)
    plt.plot(REWARDS)
    
    ax2 = plt.subplot(4,2,8)
    plt.plot(R)
    plt.plot(R_PRED)
    ax2.plot(R, label=f'real')
    ax2.plot(R_PRED, label=f'pred.')
    ax2.legend()

    plt.subplot(4,2,3)
    plt.imshow(1-np.array(S_agent)[:,0:512].T,aspect='auto',cmap ='gray')
    
    plt.subplot(4,2,5)
    plt.plot(np.array(OUT))
    plt.ylim(-.1,1.1)
    plt.ylabel('policy')


    ax = plt.subplot(4,2,4)
    ax.plot(np.array(RAM)[:,0], label=f'ball x')
    if RAM_PRED:
       ax.plot(np.array(RAM_PRED)[:,0], label=f'ball x (pred)')

    plt.subplot(4,2,7)
    x = np.linspace(50, 50*len(REWARDS_MEAN), len(REWARDS_MEAN))
    plt.plot(x, ENTROPY_MEAN)
    plt.ylabel('entropy')
    plt.xlabel('iterations')

    ax = plt.subplot(4,2,6)
    ax.plot(np.array(RAM)[:,1], label=f'ball y')
    ax.plot(np.array(RAM)[:,2], label=f'cpu y')
    ax.plot(np.array(RAM)[:,3], label=f'player y')
    if RAM_PRED:
        ax.plot(np.array(RAM_PRED)[:,1], label=f'ball y (pred)')
        ax.plot(np.array(RAM_PRED)[:,2], label=f'cpu y (pred)')
        ax.plot(np.array(RAM_PRED)[:,3], label=f'player y (pred)')

    plt.savefig(filename, facecolor='w', edgecolor='w')
    plt.close()
    return

def plot_rewards_dream(REWARDS, REWARDS_MEAN, S_agent, S_planner, OUT, RAM, RAM_PRED, R, R_PRED, ENTROPY_MEAN, filename = 'figure.png'):
    #plt.figure(figsize=(18, 18), dpi=100)
    plt.figure(figsize=(8, 11), dpi=100)
    plt.suptitle('Number of spikes: ' + str(np.sum(S_agent)) + ' (Agent), ' + str(np.sum(S_planner)) + ' (Model)', fontsize=11)

    plt.subplot(4,2,1)
    # create a list from 0 to 2000 in steps of 50
    x = np.linspace(50, 50*len(REWARDS_MEAN), len(REWARDS_MEAN))
    plt.plot(x, REWARDS_MEAN)
    plt.ylabel('reward mean')
    # plt.xlabel('iterations')

    plt.subplot(4,2,2)
    plt.plot(REWARDS)
    
    ax2 = plt.subplot(4,2,8)
    ax2.plot(R, label=f'reward')
    ax2.plot(R_PRED, label=f'(pred)')
    ax2.legend()
    
    plt.subplot(4,2,3)
    plt.imshow(1-np.array(S_agent)[:,0:512].T,aspect='auto',cmap ='gray')
    #plt.xlabel('policy')

    plt.subplot(4,2,5)
    plt.plot(np.array(OUT))
    plt.ylim(-.1,1.1)
    # plt.xlabel('time')
    plt.ylabel('policy')


    ax = plt.subplot(4,2,4)
    ax.plot(np.array(RAM)[:,0], label=f'ball x')
    # if RAM_PRED:
    #    ax.plot(np.array(RAM_PRED)[:,0], label=f'ball x (pred)')
    # plt.plot(np.array(RAM_PRED)[:,0])
    # plt.plot(np.array(RAM)[:,0])
    #plt.xlabel('step')
    #plt.ylabel('ball x')

    plt.subplot(4,2,7)
    x = np.linspace(50, 50*len(REWARDS_MEAN), len(REWARDS_MEAN))
    plt.plot(x, ENTROPY_MEAN)
    plt.ylabel('entropy')
    plt.xlabel('iterations')

    ax = plt.subplot(4,2,6)
    ax.plot(np.array(RAM)[:,1], label=f'ball y')
    ax.plot(np.array(RAM)[:,2], label=f'cpu y')
    ax.plot(np.array(RAM)[:,3], label=f'player y')
    #if RAM_PRED:
    #    ax.plot(np.array(RAM_PRED)[:,1], label=f'ball y (pred)')
    #    ax.plot(np.array(RAM_PRED)[:,2], label=f'cpu y (pred)')
    #    ax.plot(np.array(RAM_PRED)[:,3], label=f'player y (pred)')
    ax.legend()

    plt.savefig(filename, facecolor='w', edgecolor='w')
    plt.close()
    return

def plot_dram (OUT, DRAM,DRAM_PRED,R,R_PRED,MEAN_ERROR_RAM,MEAN_ERROR_R, S_planner, filename = 'figure.png'):

    plt.figure(figsize=(10,12), dpi=72)

    plt.subplot(421)
    plt.plot(np.array(DRAM_PRED)[:,0])
    plt.plot(np.array(DRAM)[:,0])
    plt.xlim(0,100)
    plt.ylabel('ball x')

    plt.subplot(422)
    plt.plot(np.array(DRAM_PRED)[:,1])
    plt.plot(np.array(DRAM)[:,1])
    plt.xlim(0,100)
    plt.ylabel('ball y')

    plt.subplot(423)
    plt.plot(np.array(DRAM_PRED)[:,2])
    plt.plot(np.array(DRAM)[:,2])
    plt.xlim(0,100)
    plt.ylabel('cpu y')

    plt.subplot(424)
    plt.plot(np.array(DRAM_PRED)[:,3])
    plt.plot(np.array(DRAM)[:,3])
    plt.xlim(0,100)
    plt.ylabel('player y')

    plt.subplot(426)
    plt.plot(np.array(R_PRED))
    plt.plot(np.array(R))
    plt.xlim(0,100)
    plt.ylabel('reward')

    ax2 = plt.subplot(425)
    x = np.linspace(50, 50*len(MEAN_ERROR_RAM), len(MEAN_ERROR_RAM))
    ax2.plot(x, np.array(MEAN_ERROR_RAM)[:,0], label=f'ball x')
    ax2.plot(x, np.array(MEAN_ERROR_RAM)[:,1], label=f'ball y')
    ax2.plot(x, np.array(MEAN_ERROR_RAM)[:,2], label=f'cpu y')
    ax2.plot(x, np.array(MEAN_ERROR_RAM)[:,3], label=f'player y')
    ax2.legend()
    # plt.plot(x, MEAN_ERROR_RAM)
    # plt.plot(MEAN_ERROR_RAM)
    # plt.plot(ERROR_RAM)
    # plt.xlabel('iteration')
    plt.ylabel('dram mean error')

    plt.subplot(427)
    x = np.linspace(50, 50*len(MEAN_ERROR_R), len(MEAN_ERROR_R))
    plt.plot(x, MEAN_ERROR_R)
    #plt.plot(MEAN_ERROR_R)
    #plt.plot(ERROR_R)
    plt.xlabel('iteration')
    plt.ylabel('reward mean error')

    plt.subplot(428)
    plt.imshow(1-np.array(S_planner)[:,0:512].T,aspect='auto',cmap ='gray')
    

    plt.savefig(filename)
    plt.close()
    return

def plot_planning(OUT, OUT_dream, REWS_PLAN,R,RAM_PLAN,RAM,S_agent,S_planner,t_skip,filename):
    #t_skip = t_skip-1
    plt.figure(figsize=(10,10), dpi=72)

    plt.subplot(5, 2, 1)
    plt.plot(np.array(RAM)[:,0])
    plt.ylabel('x (real)')
    
    ax = plt.subplot(522)
    ax.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,0], label=f'ball x')
    ax.legend()
    #plt.plot(np.array(RAM)[:,1])
    #plt.plot([0, 0],[0, 200],'r--')
    # plt.xlabel('step')
    plt.ylabel('x (dream)')
    #plt.xlim(0,100)

    plt.subplot(5, 2, 3)
    #plt.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,0])
    #plt.plot(np.array(RAM)[:,0])
    plt.plot(np.array(RAM)[:,1])
    plt.plot(np.array(RAM)[:,2])
    plt.plot(np.array(RAM)[:,3])
    #plt.plot([0, 0],[0, 200],'r--')
    # plt.xlabel('step')
    plt.ylabel('y (real)')
    #plt.ylabel('x_ball')
    #plt.xlim(0,100)
    
    ax = plt.subplot(5, 2, 4)
    ax.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,1], label=f'ball y')
    ax.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,2], label=f'cpu y')
    ax.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,3], label=f'player y')
    ax.legend()
    #plt.plot(np.array(RAM)[:,1])
    #plt.plot([0, 0],[0, 200],'r--')
    # plt.xlabel('step')
    plt.ylabel('y (dream)')
    #plt.xlim(0,100)

    plt.subplot(5, 2, 5)
    plt.plot(np.array(OUT))
    plt.ylim(-.1,1.1)
    plt.xlabel('time')
    plt.ylabel('policy')
    # plt.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,2])
    # plt.plot(np.array(RAM)[:,2])
    # plt.plot([0, 0],[0, 200],'r--')
    # # plt.xlabel('step')
    # plt.ylabel('y cpu')
    #plt.xlim(0,100)
    
    plt.subplot(5, 2, 6)
    plt.plot(np.array(OUT_dream))
    plt.ylim(-.1,1.1)
    plt.xlabel('time')
    plt.ylabel('policy')
    # plt.plot(np.arange(0, len(REWS_PLAN)), np.array(RAM_PLAN)[:,3])
    # plt.plot(np.array(RAM)[:,3])
    # plt.plot([0, 0],[0, 200],'r--')
    # # plt.xlabel('step')
    # plt.ylabel('y player')
    
    plt.subplot(5, 2, 7)
    plt.plot(np.arange(0, len(REWS_PLAN)), np.array(REWS_PLAN))
    plt.plot(R)
    #plt.plot([0, 0],[-1, 1],'r--')
    # plt.xlabel('step')
    plt.ylabel('reward')
    #plt.xlim(0,100)
    
    plt.subplot(5, 2, 9)
    plt.plot(np.array(RAM_PLAN)[:,0],np.array(REWS_PLAN),'o')
    plt.xlabel('x_ball')
    plt.ylabel('rew')
    
    plt.subplot(5, 2, 8)
    plt.imshow(1-np.array(S_agent)[:,0:512].T,aspect='auto',cmap ='gray')
    #plt.xlabel('time')
    plt.ylabel('agent net')
    
    plt.subplot(5, 2, 10)
    plt.imshow(1-np.array(S_planner)[:,0:512].T,aspect='auto',cmap ='gray')
    plt.xlabel('time')
    plt.ylabel('model net')
    
    plt.savefig(filename, facecolor='w', edgecolor='w')
    plt.close()
    return
