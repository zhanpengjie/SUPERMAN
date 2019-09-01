from pyspark.sql import SparkSession
import MyUtil,os,sys

def handle(data,actions):
	
	"""
	generate dict to include data info,
	"res" include every userid ,every userId include its productIds ,actionIds and scores
	"""
	res = {}

	"""
	every "item" in data include 5 column
        item[0] is actionId
        item[1] is productId
	item[2] is score
 	item[3] is timestamp
	item[4] is userId
	"""
	data = data.collect()
	for item in data:
        #res中如果没有该用户ID则在res中将该用户的ID赋空集合
		if item[4] not in res:
			res[item[4]] = {}
        #如果中该用户没有产品ID，则res中将该用户的产品ID赋空集合
		if item[1] not in res[item[4]]:
			res[item[4]][item[1]] = {}
        #如果该用户没有动作ID，则把打分赋值到动作ID位置
		if item[0] not in res[item[4]][item[1]]:
			res[item[4]][item[1]][item[0]] = item[2]
        #如果存在动作ID并且对应到分数与原始用户打分不同则用原始打分覆盖
		elif item[0] in res[item[4]][item[1]] and item[2] != res[item[4]][item[1]][item[0]]:
			res[item[4]][item[1]][item[0]] = item[2]
		
		elif item[0] not in res[item[4]][item[1]]:
			res[item[4]][item[1]][item[0]] = item[2]

	"""
	return result, include user and their socre towords product according to different action labels and weights
	action labels and its weight are in recommendation_conf.json 
	"""
	result = {}

	# item is userId, init result by userId
	for item in res:
		if item not in result:
			result[item] = {}
	
	# item is userId, init result by productId
	for item in res:
		for product in res[item]:
			if product not in result[item]:
				result[item][product] =''

	# item is userId, upload result by total score
	for item in res:
		for product in res[item]:

			# sum is the score that include several action-score and respective weight
			sum = 0
			for key in res[item][product].keys():
				sum+= float(actions[key]) * int(res[item][product][key])

			if str(sum) not in result[item][product] :
				result[item][product]= round(sum,2)
	"""
	重点，以上处理就是将原始数据处理成以下形式，即每个用户对应其下所有产品的评分
	其中评分按权重进行相加
	the format of result as follow:
	result = {
                userId1:{productId1:score1,productId2:score2,...},
                userId2:{productId1:score1,productId2:score2,...},
                ...
                }
	"""
	return result