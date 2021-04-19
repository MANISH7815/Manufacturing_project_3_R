
getwd()

p_train=read.csv("product_train.csv",stringsAsFactors = F)
p_test=read.csv("product_test.csv",stringsAsFactors = F)

library(dplyr)
library(tidyr)
library(ggplot2)

glimpse(p_train)
glimpse(p_test)

p_test$went_on_backorder=NA

p_train$data='train'
p_test$data='test'

p_all=rbind(p_train,p_test)

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  data[,var]=NULL
  return(data)
}

names(p_all)[sapply(p_all, function(x) is.character(x))]

table(p_all$potential_issue)

#if potential_issue=='Yes' it will put as 1
p_all$potential_issue=as.numeric(p_all$potential_issue=='Yes')

table(p_all$deck_risk)

p_all$deck_risk=as.numeric(p_all$deck_risk=="Yes")

table(p_all$oe_constraint)

p_all$oe_constraint=as.numeric(p_all$oe_constraint=='Yes')

table(p_all$ppap_risk)

p_all$ppap_risk=as.numeric(p_all$ppap_risk=="Yes")

names(p_all)[sapply(p_all,function(x) is.na(x))]

table(p_all$stop_auto_buy)

p_all$stop_auto_buy=as.numeric(p_all$stop_auto_buy=="Yes")

table(p_all$rev_stop)

p_all$rev_stop=as.numeric(p_all$rev_stop=="Yes")

glimpse(p_all)

table(p_all$went_on_backorder)

p_all$went_on_backorder=as.numeric(p_all$went_on_backorder=="Yes")

#--------------------------------------------------------------------------------------
#now filtering it 

p_train=p_all %>% 
  filter(data=="train") %>% 
  select(-data)

p_test=p_all %>%
  filter(data=="test") %>% 
  select(-data,-went_on_backorder )

#-------------------------------------------------------------------------------------

lapply(p_all,function(x) length(unique(x)))

lapply(p_all,function(x) sum(is.na(x)))

lapply(p_test,function(x) sum(is.na(x)))
lapply(p_train,function(x) sum(is.na(x)))

#-------------------------------------------------------------------------
#splitting the p_train

set.seed(3)
s=sample(1:nrow(p_train),0.8*nrow(p_train))
p_train1=p_train[s,]
p_train2=p_train[-s,]

#now starting building model

library(car)

for_vif=lm(went_on_backorder~.-sku-forecast_9_month-sales_6_month
           -sales_3_month-forecast_6_month-sales_9_month,
           data = p_train1)

sort(vif(for_vif),decreasing = T)[1:3]

log_fit=glm(went_on_backorder~.-sku-forecast_9_month-sales_6_month
            -sales_3_month-forecast_6_month-sales_9_month,
            data = p_train1,family = "binomial")

log_fit=step(log_fit)

formula(log_fit)

log_fit=glm(went_on_backorder ~ national_inv + lead_time + in_transit_qty + 
              forecast_3_month + min_bank + potential_issue + perf_12_month_avg + 
              deck_risk+ rev_stop,data = p_train1,family = "binomial")


summary(log_fit)

library(pROC)

val.score=predict(log_fit,newdata = p_train2,type='response')
auc_score=auc(roc(p_train2$went_on_backorder,val.score))

auc_score

##------------------------------------------------------------------------------------
library(ggplot2)
mydata=data.frame(went_on_backorder=p_train2$went_on_backorder,val.score=val.score)

ggplot(mydata,aes(y=went_on_backorder,x=val.score,color=factor(went_on_backorder)))+
  geom_point()+geom_jitter()

##---------------------------------------------------------------------------------------
#building on entire data

for_final_vif=lm(went_on_backorder~.-sku-forecast_9_month-sales_6_month
           -sales_3_month-forecast_6_month-sales_9_month,
           data = p_train)

sort(vif(for_final_vif),decreasing = T)[1:3]

log_final_fit=glm(went_on_backorder~.-sku-forecast_9_month-sales_6_month
            -sales_3_month-forecast_6_month-sales_9_month,
            data = p_train,family = "binomial")


log_final_fit=step(log_final_fit)


log_final_fit=glm(went_on_backorder~.-sku-forecast_9_month-sales_6_month
                  -sales_3_month-forecast_6_month-sales_9_month,
                  data = p_train,family = "binomial")

summary(log_final_fit)


formula(log_final_fit)

log_final_fit=glm(went_on_backorder ~ sku + national_inv + lead_time + 
                    in_transit_qty + forecast_6_month + 
                    forecast_9_month +sales_1_month + sales_3_month + sales_6_month + 
                    sales_9_month +  min_bank + potential_issue + pieces_past_due +
                    perf_6_month_avg  + local_bo_qty + deck_risk  
                    + rev_stop, 
                  data = p_train,family = "binomial")


summary(log_final_fit)

test.prob.score= predict(log_final_fit,newdata = p_test,type='response')

write.csv(test.prob.score,"firstName_LastName_P3_part2_1.csv",row.names = F)
##-----------------------------------------------------------------------------------------


train.score=predict(log_final_fit,newdata = p_train,type='response')

real=p_train$went_on_backorder

cutoffs=seq(0.001,0.999,0.0001)

cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)


for(cutoff in cutoffs){
  
  predicted=as.numeric(train.score>cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  
  N=TN+FP
  
  Sn=TP/P
  
  Sp=TN/N
  
  precision=TP/(TP+FP)
  
  recall=Sn
  
  KS=(TP/P)-(FP/N)
  
  F5=(26*precision*recall)/((25*precision)+recall)
  
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data=cutoff_data[-1,]

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff


# c1, max KS
c1=cutoff_data$cutoff[which.max(cutoff_data$KS)]
c1
max(cutoff_data$KS)

# c2 max F_0.1
c2=cutoff_data$cutoff[which.max(cutoff_data$F.1)]
c2
max(cutoff_data$F.1)

# c3 max F_3
c3=cutoff_data$cutoff[which.max(cutoff_data$F5)]
c3
max(cutoff_data$F5)

library(ggplot2)
ggplot(cutoff_data,aes(x=cutoff,y=Sp))+geom_line()

library(tidyr)
cutoff_long=cutoff_data %>%
  gather(Measure,Value,Sn:M)

ggplot(cutoff_long,aes(x=cutoff,y=Value,color=Measure))+geom_line()

am=1-(0.025/0.059)

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff

test.predicted=as.numeric(test.prob.score>my_cutoff)

table(test.predicted)

write.csv(test.predicted,"proper_submission_file_name.csv",row.names = F)


pt=read.csv("proper_submission_file_name.csv",stringsAsFactors = F)

pt=pt %>% 
  mutate(x=ifelse(x==1,"Yes","No"))

write.csv(pt,"firstName_LastName_P3_part2.csv",row.names = F)










