##Application
this project is used to handle data from different data source(hdfs or hive ) by FP-Growth, ALS, Kmeans three algorighoms,then recommendate productions for every user.

##Data 
the data format as follows:
		
	actionId	productId	score	timestamp	userid
	
	4			6			0		2018/9/18	1
	1			4			1		2018/9/18	1
	3			2			5		2018/9/18	1
	2			10			5		2018/9/18	1
	1			3			0		2018/9/18	2
	2			2			4		2018/9/18	2
	...
	
The actions 1-4 are assumed to be these as follows:
	
	1->"点赞"
	2->"打分"
	3-> "转发"
	4->"浏览"

##Show default user's info 、product's info and product's total score


	PYSPARK=1 PYSPARK_PROGRAM=/home/fengpeng/loadInfo.py SPARK_PROGRAM_PARAMS="hdfs /user/fengpeng/product_score.csv 3 /home/fengpeng/recommendation_conf.json /user/fengpeng"  bash /spark-2.3.0-bin-hadoop2.7/execute.sh
	
	"/home/fengpeng/loadInfo.py" is the drive file path.
	"hdfs" is the data input mode.
	"user/fengpeng/product_score.csv" is the data location.
	"3" is the userId.
	"/home/fengpeng/recommendation_conf.json" is the conf file path.
	"/user/fengpeng" is the output path.
	
result is as follows:

	============User Info================== //actions User performed 
	{u'3': 1, u'2': 2, u'4': 1} 
	============Product Info============== // detailed number of action for each product
	{u'10': {u'1': 107, u'3': 91, u'2': 112, u'4': 97}, u'1': {u'1': 90, u'3': 93, u'2': 85, u'4': 86}, u'3': {u'1': 109, u'3': 103, u'2': 117, u'4': 95}, u'2': {u'1': 75, u'3': 119, u'2': 113, u'4': 114}, u'5': {u'1': 117, u'3': 88, u'2': 88, u'4': 95}, u'4': {u'1': 104, u'3': 101, u'2': 86, u'4': 110}, u'7': {u'1': 93, u'3': 88, u'2': 109, u'4': 100}, u'6': {u'1': 88, u'3': 102, u'2': 97, u'4': 121}, u'9': {u'1': 108, u'3': 97, u'2': 103, u'4': 86}, u'8': {u'1': 103, u'3': 99, u'2': 112, u'4': 99}}
	============Product Score============== //total score for each product （Descending）
	[(u'5', 330.0999999999999), (u'10', 308.4999999999999), (u'8', 308.49999999999983), (u'4', 306.4000000000002), (u'9', 297.3999999999999), (u'3', 296.89999999999975), (u'7', 284.6999999999999), (u'6', 278.50000000000017), (u'1', 252.79999999999998), (u'2', 247.4999999999996)]
	
save above output to generate some files those save different algorithm result,as follows:
	
	/user/fengpeng/productScore.csv
	/user/fengpeng/productInfo.json
	
##Ready
set properties what you want in the **recommendation_conf.json**,or you can use default values.

If the weight of each action needs to be changed, you can set it，but the weight's sum of all actions is 1,as follows:

	"actions":{
        "点赞":0.4（分值为0表示没有点赞行为，分值为5表示有点赞行为），
        "打分":0.3（打分范围在整数1-5分之间）,
        "转发":0.2（分值为0表示没有转发行为，分值为5表示有转发行为）,
        "浏览":0.1（分值为0表示没有浏览行为，分值为5表示有浏览行为）
    }


## Run this command in pyspark cluster and save result to hfds
if the data is from **hdfs**,run command as follows:

 	PYSPARK=1 PYSPARK_PROGRAM=/home/fengpeng/MAIN.py SPARK_PROGRAM_PARAMS="/home/fengpeng/recommendation_conf.json hdfs /user/fengpeng/product_score.csv /user/fengpeng ALS"  bash /spark-2.3.0-bin-hadoop2.7/execute.sh
 	 	
 	"/home/fengpeng/MAIN.py" is the drive file path.
	"/home/fengpeng/recommendation_conf.json" is the conf file path.
	"hdfs" is the data input mode.
	"/user/fengpeng/product_score.csv" is the data location in hdfs.
	"/user/fengpeng" is the output path.
	"ALS" is the algorithm.

generate some files those save different algorithm result,as follows:

	/user/fengpeng/ALSresult.json
	/user/fengpeng/FPresult.json
	/user/fengpeng/KMEANSresult.json
	/user/fengpeng/FPmodel
 
_____________
if the data is from **hive**,run command as follows:

	PYSPARK=1 PYSPARK_PROGRAM=/home/fengpeng/MAIN.py SPARK_PROGRAM_PARAMS="/home/fengpeng/recommendation_conf.json hive fengpeng.product_score /user/fengpeng ALS"  bash /spark-2.3.0-bin-hadoop2.7/execute.sh
	 	
 	"/home/fengpeng/MAIN.py" is the drive file path.
	"/home/fengpeng/recommendation_conf.json" is the conf file path.
	"hive" is the data input mode.
	"fengpeng.product_score" is the table location in hive.
	"/user/fengpeng" is the output path.
	"ALS" is the algorithm.
generate the same result file as above.
	
	
##Just run this command in pyspark cluster to interact with user


	PYSPARK=1 PYSPARK_PROGRAM=/home/fengpeng/interactWithUser.py SPARK_PROGRAM_PARAMS="/user/fengpeng/product_score.csv 30 FP /user/fengpeng 2 3 1"  bash /spark-2.3.0-bin-hadoop2.7/execute.sh
	
	"/home/fengpeng/interactWithUser.py" is the drive file path.
	"user/fengpeng/product_score.csv" is the data location.
	"30" is the userId.
	"FP" is the algorithm.
	"/user/fengpeng" is the model save location.
	"2" means that results found by action 2.
	"3" is the number of output.
	"1" means that results sorted by descending (0 means Ascending).
the recommendation result as follows:
	
	============FP==============
	[[u'6', 0.16071428571428573], [u'4', 0.15476190476190477], [u'2', 0.14285714285714285], [u'5', 0.13690476190476192], [u'8', 0.13095238095238096], [u'3', 0.13095238095238096], [u'7', 0.11904761904761904]]
	===filter by action 2=======
	[(u'3', 117), (u'2', 113), (u'8', 112)]  //(u'3', 117) means productid 3 has been executed 117 times by action 2，Others and so on
	
	