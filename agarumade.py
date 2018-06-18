import random
import re
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils import plot_model
from keras import losses
from collections import deque
import tensorflow as tf
from copy import deepcopy
from datetime import datetime
from time import time

shanten = Shanten()
pailist =[]
tehailist = []
senpai = []
result = 0
syanten = 0

class Mahjong():
    ripaitehai = []
    def pais(self):
        global senpai

        for num in range(1, 10):
            pai = str(num) + 'm'
            for n in range(4):
                pailist.append(pai)

        for num in range(1, 10):
            pai = str(num) + 'p'
            for n in range(4):
                pailist.append(pai)

        for num in range(1, 10):
            pai = str(num) + 's'
            for n in range(4):
                pailist.append(pai)

        """
        for num in range(1, 8):
            pai = str(num) + 'z'
            for n in range(4):
                pailist.append(pai)
        """

    def yamatumi(self):
        returnyama = deque()
        senpai = random.sample(pailist, len(pailist))
        for i in senpai:
            returnyama.append(i)
        return returnyama

    def haipai(self, yama):
        tehai = [yama.popleft() for i in range(0, 13)]
        return tehai, yama

    def tumo(self, tehai, yama):
        resultfile.write('ツモ：' + str(yama[0]) + '\n')
        tehai.append(yama.popleft())

        return tehai, yama

    def dahai(self, tehai, action, kawa):
        resultfile.write('打：' + str(tehai[action]) + '\n\n')
        kawa.append(tehai.pop(action))

        return tehai, kawa

    def randomdahai(self):
        kawa.append(tehai.pop(random.randint(0, 13)))

    def ripai(self, tile):
        cpytehai = []
        man = []
        pin = []
        sou = []
        honors = []

        cpytehai.extend(tile)
        for n in range(len(cpytehai)):
            pai = cpytehai.pop()
            if 'm' in str(pai):
                man.append(pai)
            elif 'p' in str(pai):
                pin.append(pai)
            elif 's' in str(pai):
                sou.append(pai)
            else:
                honors.append(pai)

        result = sorted(man) + sorted(pin) + sorted(sou) + sorted(honors)

        return result, man, pin, sou, honors

    def maketehailist(self, tile):
        man2 = []
        pin2 = []
        sou2 = []
        honors2 = []

        __, man1, pin1, sou1, honors1 = mahjong.ripai(tile)


        for a in range(len(man1)):
            man2.append(re.sub('m', '', man1.pop()))
        for a in range(len(pin1)):
            pin2.append(re.sub('p', '', pin1.pop()))
        for a in range(len(sou1)):
            sou2.append(re.sub('s', '', sou1.pop()))
        for a in range(len(honors1)):
            honors2.append(re.sub('z', '', honors1.pop()))

        tehailist = mahjong.henkan(man2, pin2, sou2, honors2)

        return tehailist

    def syanten(self, tehais):
        result = shanten.calculate_shanten(tehais)

        return result

    def henkan(self, man, pin, sou, jihai):
        m = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        p = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        s = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        z = [0, 0, 0, 0, 0, 0, 0]

        list(map(int, man))
        list(map(int, pin))
        list(map(int, sou))
        list(map(int, jihai))

        for i in range(0, len(man)):
            nm = man.pop()
            nm = int(nm)
            m[nm - 1] = m[nm - 1] + 1

        for i in range(0, len(pin)):
            np = pin.pop()
            np = int(np)
            p[np - 1] = p[np - 1] + 1

        for i in range(0, len(sou)):
            ns = sou.pop()
            ns = int(ns)
            s[ns - 1] = s[ns - 1] + 1

        for i in range(0, len(jihai)):
            nz = jihai.pop()
            nz = int(nz)
            z[nz - 1] = z[nz - 1] + 1

        total = m + p + s + z

        return total

class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=34, action_size=1, hidden_size=25):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model = multi_gpu_model(self.model, gpus=2)
        self.model.compile(loss=losses.mean_absolute_error, optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 34))
        targets = np.zeros((batch_size, 1))
        mini_batch = memory.sample(batch_size)
        targetmatome = []

        for i, (state_b, reward_b, next_state_b, haitei) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b
            max = -100
            b = 0
            j = 0

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）

                while b < 34:
                    if type(next_state_b) != list:
                        cpystate = np.ndarray.tolist(next_state_b)[0]
                    else:
                        cpystate = deepcopy(next_state_b)

                    if cpystate[b] != 0:
                        cpystate[b] -= 1
                        irerustate = np.reshape(cpystate, [1, 34])

                        action = actor.get_action(irerustate, targetQN)  # 時刻tでの行動を決定する

                        if max < action:
                            max = action
                            j = b
                        cpystate[b] += 1
                        b += 1
                    else:
                        b += 1
                next_state_b[0][j] -= 1

                if haitei == 1:
                    targetmatome.append(reward_b)
                else:
                    targetmatome.append(reward_b + gamma * max)
                next_state_b[0][j] += 1

        targets = self.model.predict(inputs)    # Qネットワークの出力
        for i in range(len(targetmatome)):
           targets[i] = targetmatome[i]               # 教師信号
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


# [4]カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, targetQN):   # [C]ｔ＋１での行動を返す

        retTargetQs = targetQN.model.predict(state)[0]

        return retTargetQs


# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
DQN_MODE = 0    # 1がDQN、0がDDQNです

num_episodes = 5000# 総試行回数
max_number_of_steps = 50  # 1試行のstep数
goal_average_reward = 80  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_episodes)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数

# ---
hidden_size = 25              # Q-networkの隠れ層のニューロンの数
learning_rate = 0.00001         # Q-networkの学習係数
memory_size = 100000000            # バッファーメモリの大きさ
batch_size = 100              # Q-networkを更新するバッチの大記載

# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
#plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory = Memory(max_size=memory_size)
actor = Actor()
mahjong = Mahjong()
mahjong.pais()

tenpai_count = 0
heikinjunme = 0

resultfile = open('result' + datetime.now().strftime("%m%d %H%M") + '.txt', 'w')

# [5.3]メインルーチン--------------------------------------------------------
for episode in range(num_episodes):  # 試行数分繰り返す
    yama = deque()
    tehai = []
    tehailist = []
    kawa = []
    print('東' + str(episode+1) + '局')
    resultfile.write('東' + str(episode+1) + '局\n')
    i = 0
    reward = 0
    haitei = 0
    state = []
    cpystate = []
    irerustate = np.zeros((1, 34))
    epsilon = 0.001 + 0.9 / (1.0 + episode)
    done = False
    yama = mahjong.yamatumi()
    tehai, yama = mahjong.haipai(yama)
    resultfile.write('配牌' + str(tehai) + '\n')
    tehai, yama = mahjong.tumo(tehai, yama)
    backsyanten, _, _ = shanten.calculate_shanten(mahjong.maketehailist(tehai))
    state = mahjong.maketehailist(tehai)
    episode_reward = 0

    targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
    for t in range(18):  # 1試行のループ
        max = -100
        action = -100
        i = 0
        j = 0
        count = 0

        resultfile.write('手牌' + str(tehai) + '\n')

        if type(state) != list:
            cpystate = np.ndarray.tolist(state)[0]
        else:
            cpystate = deepcopy(state)
        while i < 34:
            if cpystate[i] > 0:
                cpystate[i] -= 1
                irerustate = np.reshape(cpystate, [1, 34])

                action = actor.get_action(irerustate, mainQN)   # 時刻tでの行動を決定する

                if max < action:
                    max = action
                    j = i
                cpystate[i] += 1
                i += 1
            else:
                i += 1

        if type(state) != list:
            countstate = np.ndarray.tolist(state)[0]
        else:
            countstate = state[:]
        for z in range(0, j):
            if countstate[z] != 0 and countstate[z] != -1:
                count += cpystate[z]

        if count == -1:
            print(j)
            print(count)

        tehai, _, _, _, _ = mahjong.ripai(tehai)
        if epsilon <= np.random.uniform(0, 1):
            tehai, kawa = mahjong.dahai(tehai, count, kawa)
        else:
            tehai, kawa = mahjong.dahai(tehai, random.randint(0, 13), kawa)
        syanten, _, _ = shanten.calculate_shanten(mahjong.maketehailist(tehai))
        state = np.reshape(mahjong.maketehailist(tehai), [1, 34])[0]
        resultfile.write(str(syanten) + 'シャンテン' + '\n')

        if syanten < backsyanten:
            reward = 1
        else:
            reward = 0

        if syanten == 0:
            done = True
            reward = 100
            haitei = 1
            resultfile.write('テンパった！！！\n')
            print('テンパった！！！！')
            print(tehai)
        if t == 17:
            haitei = 1
            if syanten != 0:
                reward = -100

        resultfile.write('報酬：' + str(reward) + '\n')

        tehai, yama = mahjong.tumo(tehai, yama)

        tehai, _, _, _, _ = mahjong.ripai(tehai)

        state = np.reshape(state, [1, 34])
        next_state = np.reshape(mahjong.maketehailist(tehai), [1, 34])  # list型のstateを、1行4列の行列に変換

        episode_reward += reward  # 合計報酬を更新]
        memory.add((deepcopy(state), reward, deepcopy(next_state), haitei))     # メモリの更新する
        state = next_state[:]  # 状態更新
        backsyanten = syanten

        ######

        if len(yama) <= 14:
            done = True
        # Qネットワークの重みを学習・更新する replay
        if (memory.len() > batch_size * 10):
            mainQN.replay(memory, batch_size, gamma, targetQN)

        total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録

        if syanten == 0:

            print(str(t + 1) + '順目')
            tehai, _, _, _, _ = mahjong.ripai(tehai)
            tenpai_count += 1
            heikinjunme += (t + 1)
            break

    print('mean %f' % (total_reward_vec.mean()))
    print('流局')

print(str(tenpai_count)+'回テンパった')
if tenpai_count != 0:
    print('平均聴牌順目は'+str(heikinjunme / tenpai_count))

resultfile.close()