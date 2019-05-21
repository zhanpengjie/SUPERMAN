from sklearn.datasets import kddcup99
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam
from keras.utils import to_categorical  #onehot编码
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier


#定义kdd99数据预处理函数
def preHandel_data(data_source):
    data = data_source;
    for i in range(data_source.shape[0]):
        row = data_source[i];            #获取数据
        data[i][1]=handleProtocol(row)   #将源文件行中3种协议类型转换成数字标识
        data[i][2]=handleService(row)    #将源文件行中70种网络服务类型转换成数字标识
        data[i][3]=handleFlag(row)       #将源文件行中11种网络连接状态转换成数字标识
    return data;
 
def preHandel_target(target_data_source):
    target = target_data_source;
    for i in range(target_data_source.shape[0]):
        row = target_data_source[i];
        target[i]=handleLabel(row)        #将源文件行中23种攻击类型转换成数字标识
    return to_categorical(target);
 
#定义将源文件行中3种协议类型转换成数字标识的函数
def handleProtocol(input):
    protocol_list=['tcp','udp','icmp']
    tmp = bytes.decode(input[1])
    if tmp in protocol_list:
        return protocol_list.index(tmp);
 
#定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleService(input):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50'];
    tmp = bytes.decode(input[2]);
    if tmp in service_list:
        return service_list.index(tmp);
 
#定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    tmp = bytes.decode(input[3])
    if tmp in flag_list:
        return flag_list.index(tmp);
 
#定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
def handleLabel(label):
    label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
     'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
     'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
     'spy.', 'rootkit.']
    tmp = bytes.decode(label)
    if tmp in label_list:
        return label_list.index(tmp)
 
if __name__ == '__main__':
    # 下载数据集
    dataset = kddcup99.fetch_kddcup99();    
    data = dataset.data;
    label = dataset.target;
    #print(data)
    # 数据预处理
    data = preHandel_data(data);
    label = preHandel_target(label);                #进行OneHot编码    
    input_data_train_set = data[0:300000];
    target_data_train_set = label[0:300000];
    input_data_test_set = data[300000::];
    target_data_test_set = label[300000::]; 
    #print(data[0:20])   
    
    def create_model():                      # 建立顺序神经网络层次模型
         model = Sequential();  
         model.add(Dense(40, input_dim=41, kernel_initializer='uniform', activation='relu'));#全连接层40个节点 kernel_initializer权重规范化函数
         model.add(Dense(40, kernel_initializer='uniform', activation='relu'));    # input_dim第一层形状 activation激活函数
         model.add(Dense(23, kernel_initializer='uniform', activation='softmax'));      #通过softmax函数将结果映射到23维空间
    
         rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
         sgd = SGD(lr=0.01) #优化器学习率-梯度下降
         adagrad=Adagrad(lr=0.01, epsilon=None, decay=0.0)
         adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
         adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
         adamax=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
         nadam=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

         model.compile(loss='mse', optimizer=adagrad, metrics=['accuracy']);
         #model.fit(input_data_train_set, target_data_train_set, epochs=12, batch_size=128);#epochs训练轮次 batch_size每次梯度更新的样本数
         return model
    
    model = KerasClassifier(build_fn=create_model, verbose=0)
    batch_size = [16]
    epochs = [10,20,30]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,cv=3)
    grid_result = grid.fit(input_data_train_set, target_data_train_set)
   # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
          print("%f (%f) with: %r" % (mean, stdev, param))
    
    
    #for params, mean_score, scores in grid_result.cv_results_.items():
           #print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    # 将测试集输入到训练好的模型中，查看测试集的误差
    #loss,accuracy = model.evaluate(input_data_test_set, target_data_test_set, batch_size=128);    
    #print(loss,accuracy);
    
    # 神经网络可视化
    #plot_model(model, to_file='model.png',show_shapes=True)
