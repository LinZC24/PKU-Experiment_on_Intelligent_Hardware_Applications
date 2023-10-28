from pyswip import Prolog, registerForeign
animal_dic = {} #创建空字典，用于存储规则
prolog =Prolog()
def verify(s):
    if s in animal_dic: # 若已有答案则直接返回
        return animal_dic[s]
    else:
        r = input(f"Does the animal have the following attribute: {s}?")
        # 对描述的问题进行回答
        if r.lower() in ['yes', 'y']:
            animal_dic[s] = True
            return True
        elif r.lower() in ['no', 'n']:
            animal_dic[s] = False
            return False
        else:
            print('please enter the answer \'yes\' or \'no\'')
            return verify(s)
verify.arity = 1 
registerForeign(verify) # 注册函数
prolog.consult('animal.pl') # 导入规则库
for result in prolog.query('hypothesize(X)'):
    print("I guess the animal is:", result["X"])