import pickle
from envs.garnet import Garnet


# 创建Garnet实例
garnet_instance = Garnet(S_n=3, A_n=2)

# 存储Garnet实例
with open('envs/garnet_instance.pkl', 'wb') as file:
    pickle.dump(garnet_instance, file)

print("Garnet instance created and stored.")

# 从文件中加载Garnet实例
with open('envs/garnet_instance.pkl', 'rb') as file:
    loaded_garnet = pickle.load(file)

print("Garnet instance loaded.")
print("Number of States:", loaded_garnet.S_n)
print("Number of Actions:", loaded_garnet.A_n)

# 打印一些基本信息以验证
print("Sample Transition Probabilities for state 0 and action 0:", loaded_garnet.get_transition_probabilities()[0][0])
print("Sample Rewards for state 0 and action 0:", loaded_garnet.get_rewards()[0][0])
