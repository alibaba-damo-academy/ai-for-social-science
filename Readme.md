<a name="zp9mH"></a>
# AI for Social Science
###### Summary 
This is the project of the AI for Social Science. This demo denotes a MARL framework which provides multiple auction based multi-agent environments based on Petting-zoo. We illustate multiply classical theoretical results from "Auction Theory".
Others interesting results can be explored through this platform. 
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/229273/1673851585466-b954771e-8ce9-41cf-bf4c-c4b441514a1b.png#clientId=u2af8f200-9487-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=606&id=u87b498b3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=606&originWidth=1088&originalType=binary&ratio=1&rotation=0&showTitle=false&size=133836&status=done&style=none&taskId=u828e4b11-b125-424b-94ec-7e9489df13f&title=&width=1088)

---

######  Setup & Simple Demo
This demo is based on Pettingzoo and support multiple existing RL framework such as Rllib and Tianshou.
```python
# download from the gitlab 
git clone git@github.com:alibaba-damo-academy/ai-for-social-science.git 

# then install requirements 
pip install pettingzoo

# if you want to apply deep rl in the examples 
pip install pytorch,rllib,tianshou

# then you can directly take the examples from the scripts 
# the main file is the auction_bidding_simulate.py 
# the dynamic env file is the auction_bidding_simulate_multiple.py

python auction_bidding_simulate_multiple.py --mechanism 'second_price' --exp_id 33  --folder_name 'deep_test_multi_env' \
--bidding_range 10 --valuation_range 10 --env_iters 1000000 --overbid True \
--round 1 \
--estimate_frequent 100000 --revenue_averaged_stamp 1000 --exploration_epoch 100000 --player_num 5 \
--step_floor 10000 \
```

Other demo:
For example, in the first price auction, symmetric bidders will learn their best bidding policy where there also exists a theorical bidding strategy when bidders realize the numbers of bidders and their valuation range. We validate such equilibrium when bidders only acquire their rewards without other information. 
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/229273/1673851880349-471d0eac-5190-4cba-a32e-c286855bbe6d.png#clientId=u2af8f200-9487-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=94&id=rPg4E&margin=%5Bobject%20Object%5D&name=image.png&originHeight=94&originWidth=380&originalType=binary&ratio=1&rotation=0&showTitle=false&size=18571&status=done&style=none&taskId=ub3203428-b26d-468e-9578-0ae0aaad46e&title=&width=380)

```python
# Detailed examples follows：

cd scripts/
cd same_valuation/
sh first_price.sh
```



---

###### Customed Stastic Environment

This projects offers customed environment such as signal based auction game or other complex equilibrium learning environment. We provide the platform mechanism in the file "/env/customed_env*.py" as well as the payment rule in "/env/payment_rule.py" and the allocation rule in the "/env/allocation_rule.py". 

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/229273/1673852152610-e387396e-e3fe-4c5a-9168-d4efba4fab4f.png#clientId=u2af8f200-9487-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=522&id=u82ba0a37&margin=%5Bobject%20Object%5D&name=image.png&originHeight=522&originWidth=604&originalType=binary&ratio=1&rotation=0&showTitle=false&size=47199&status=done&style=none&taskId=u0524eabe-9411-4229-99c9-fc4e006597d&title=&width=604)

---

###### More complex env design 
Ideally, agents may learn their best response through multiple learning epochs with a certian batchsize without cost ( or inital their budget during sampling) . However, in the real world, agents have to learn their best policy during the game and the whole environments evolutes when each agents may adjust their policies when the social planner changes their mechanism or information rules. 

We also provide such evolutionary envs in the “/env/multi_dynamic_env.py" and the framework illustration is denoted as below.

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/229273/1673852478564-2f3d12e0-7a08-4e95-9090-fbe09aded03a.png#clientId=u2af8f200-9487-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=513&id=u473c118d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=513&originWidth=598&originalType=binary&ratio=1&rotation=0&showTitle=false&size=54724&status=done&style=none&taskId=uede9a6c7-1cf1-4607-837f-d77b384fbef&title=&width=598)

Detailed demo can be found in the following scripts.
```python
cd scripts 
cd multi_round

sh second_price.sh
```

---

###### Apply Deep_RL lib in the framework 

We also support different RL algorithm in the agent.algorithm, such as deep RL or directly apply RL-lib.
The detailed algorithm definition is displayed in "/agent/agent_generate.py" and "rl_utils/".

As for RLlib, see examples in the "/examples/rllib_deep.py" or the "scripts /rllib_examples /rllib_examples.sh"

As for Tianshou, see examples in the "rl_utils/deep_sovler.py"

---


