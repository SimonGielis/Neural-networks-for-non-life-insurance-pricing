#This file gives a quick preview of the code used for the thesis.
#It contains: installing of keras and tensorflow, setting a seed, constructing neural networks and CANNs, gamma loss function (without weights).
#Data is not yet used in this example, the code is just meant to give an idea of the earlier described concepts. 
#The remaining code used for the thesis will be made avialable later.

install.packages("keras")
install_keras()
install.packages("tensorflow")
library(keras)
library(tensorflow)



#Set a seed for keras and for R. 
#Some functions are overwritten here because recent updates of keras, R and Python do not use the same function names.
set.seed(104) #set R seed
#Change function names to make R and python compatable.
tf$reset_default_graph <- tf$compat$v1$reset_default_graph
tf$ConfigProto<-tf$compat$v1$ConfigProto
tf$set_random_seed<-tf$compat$v1$set_random_seed
tf$Session<-tf$compat$v1$Session
tf$get_default_graph <- tf$compat$v1$get_default_graph
tf$keras$backend$set_session<-tf$compat$v1$keras$backend$set_session
#Set keras seed.
use_session_with_seed(104)

#Construct the gamma loss function.
lossgamma<- function(y_true, y_pred){
  K<- backend()
  
  
  loss <- 2*K$mean ( ( ((y_true - y_pred)/y_pred)-K$log(y_true/y_pred) ))
  loss
}
#Transform the metric into a metric usable by keras
metric_gamma <- custom_metric("lossgamma", function(y_true, y_pred) {
  lossgamma(y_true, y_pred)
})

#Build a neural network
library(keras) 
library(tensorflow)
Neural_Network<-keras_model_sequential()

Neural_Network %>% layer_dense(
  units = 20, input_shape = 11, name = "last_hidden_layer") %>%
  layer_activation(activation = "relu") %>%
  layer_dense(units = 1, activation = 'exponential') 

Neural_Network %>% compile(loss = lossgamma, optimizer = "sgd",metric=metric_gamma)

earlystopping <- callback_early_stopping(
  monitor = "val_loss", patience = 20)

#Training of neural network (code to generate covariates_train_data, train_target, covariates_test_data, test_target will be added later (MTPL data should be preprocessed)).
#The goal of this code is just to give an idea how the neural networks and CANNs are constructed.
#Later neural networks/CANNs could be trained similarly.
history <- Neural_Network %>%
  fit(covariates_train_data, train_target, epochs = 500, 
      callbacks = list(earlystopping), 
      validation_data = list(covariates_test_data, test_target))



#Building fixed and flexible CANNs
#Construct the input nodes of the continues risk factors 
Ageph<-layer_input(shape=c(1),dtype='float32',name='Ageph')
Bm<-layer_input(shape=c(1),dtype='float32',name='Bm')
Agec<-layer_input(shape=c(1),dtype='float32',name='Agec')
Power<-layer_input(shape=c(1),dtype='float32',name='Power')
Long<-layer_input(shape=c(1),dtype='float32',name='Long')
Lat<-layer_input(shape=c(1),dtype='float32',name='Lat')


Logpred<-layer_input(shape=c(1),dtype='float32',name='Logpred')
Coverage<-layer_input(shape=c(1),dtype='int32',name='Coverage')
Sex<-layer_input(shape=c(1),dtype='int32',name='Sex')  
Fuel<-layer_input(shape=c(1),dtype='int32',name='Fuel')  
Use<-layer_input(shape=c(1),dtype='int32',name='Use')
Fleet<-layer_input(shape=c(1),dtype='float32',name='Fleet')

#Make the layer embedding for the categorical risk factors.
d=1
CovEmb<-Coverage%>%
  layer_embedding(input_dim=3, output_dim=d, input_length=1, name='CovEmb')%>%
  layer_flatten(name='Cov_flat')

SexEmb<-Sex%>%
  layer_embedding(input_dim=2, output_dim=d, input_length=1, name='SexEmb')%>%
  layer_flatten(name='Sex_flat')

FuelEmb<-Fuel%>%
  layer_embedding(input_dim=2, output_dim=d, input_length=1, name='FuelEmb')%>%
  layer_flatten(name='Fuel_flat')

UseEmb<-Use%>%
  layer_embedding(input_dim=2, output_dim=d, input_length=1, name='UseEmb')%>%
  layer_flatten(name='Use_flat')

FleEmb<-Fleet%>%
  layer_embedding(input_dim=2, output_dim=d, input_length=1, name='FleEmb')%>%
  layer_flatten(name='Fle_flat')



AdjNetwork=list(Ageph,Bm,Agec,Power,Long,Lat,CovEmb,SexEmb,
                FuelEmb,UseEmb,FleEmb) %>% layer_concatenate %>%
  layer_dense(units = 20, input_shape = 11) %>%
  layer_activation(activation = "relu") %>%
  layer_dense(units=1, activation = "linear", name = 'AdjNetwork')

# Combining network for fixed CANN.
CombNetwork= list(AdjNetwork, Logpred) %>% layer_concatenate %>%
  layer_dense(units=1, activation=k_exp, name = 'CombNetwork',
              trainable = FALSE, 
              weights = list(array(1, dim = c(2,1)), array(0, dim = c(1))))

# Combining network for flexible CANN.
CombNetwork = list(AdjNetwork, Logpred) %>% layer_concatenate %>%
  layer_dense(units=1, activation = k_exp,name = 'CombNetwork',
              trainable = TRUE)

CANN <- keras_model(inputs = c(Ageph,Bm,Agec,Power,Long,Lat,
                               Coverage,Sex,Fuel,Use,Fleet,Logpred), outputs = c(CombNetwork))


#Construction of a bias regulated neural networks:
#Extract the values of the last hidden layer.
intermediate_model <- keras_model(inputs = Neural_Network$input,
                                  outputs = get_layer(Neural_Network, 'last_hidden_layer')$output)
output_last_hidden_layer <- predict(intermediate_model, covariates_train_data) 

# Use the node values of the last hidden layer as input for a glm.
Neural_Network_bias_regulated <- glm(train_target ~ output_last_hidden_layer+offset(log(exposure)),  
                                     family = poisson("log"))

#Construction of a bias regulated CANN:
#Extract the value of the output node of the adjusting network.
intermediate_model <- keras_model(inputs = CANN$input,
                                  outputs = get_layer(CANN, 'AdjNetwork')$output)
proposed_adjustment <- intermediate_model(list(ageph,bm,agec,power,long,lat,cov,sex,fuel,use,fleet,Logpred))
#ageph contains the values for the Ageph input node, similar for other data and input nodes.
#Use a glm with as input the proposed adjustment of the adjusting network, and the log(initial prediction).
CANN_bias_regulated <- glm(train_target ~ proposed_adjustment + Logpred,  
                           family= poisson("log")) 




