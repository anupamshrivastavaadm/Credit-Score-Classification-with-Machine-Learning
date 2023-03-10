# Credit-Score-Classification-with-Machine-Learning

Credit Score Classification: 
Banks and credit card issuers categorise their clients using one of three credit scores:

1. Good
2. Standard
3. Poor

Any bank or financial organisation will lend money to someone with a good credit score. We require a labelled dataset with credit scores in order to complete the task of credit score classification.

For this purpose, I discovered the perfect dataset, which was labelled with credit card users' credit histories. Download the dataset from this page.
Dataset : https://www.kaggle.com/datasets/parisrohan/credit-score-classification  

1.Importing the essential Python modules and the dataset will allow you to begin the task of credit score classification.

![image](https://user-images.githubusercontent.com/118778677/224305495-3002897a-3230-4d3b-9be2-7ed11a854822.png)

2. Let's examine the details of the columns in the dataset:

![image](https://user-images.githubusercontent.com/118778677/224305933-5bfaf51f-f78f-4f88-8970-4d39046feef2.png)

3. Before continuing, let's check to see if the dataset contains any null values:

![image](https://user-images.githubusercontent.com/118778677/224305989-f7bef6dc-162c-47ef-b2df-f61ae0ffe772.png)

4. There are no null values in the dataset. Let's have a look at the values in the Credit Score column since this dataset is labelled:

![image](https://user-images.githubusercontent.com/118778677/224306111-a3189deb-2327-4aa9-b4af-3982723b6a1b.png)

5. Exploration of Data: 
Many features in the dataset can be used to build a machine learning model for classifying credit scores. Let's examine each aspect individually.

I'll start by analyzing the occupation characteristic to determine whether a person's job has an impact on credit scores:

![newplot](https://user-images.githubusercontent.com/118778677/224306331-39748463-f297-4c23-b245-10ec8bccd310.png)

6. The credit scores of all the jobs stated in the data do not really differ all that much. Let's now explore whether or not the person's annual income affects your credit scores:

![newplot (1)](https://user-images.githubusercontent.com/118778677/224306421-d6a3a3be-dd74-4e06-a1dd-ee15b5ac7384.png)

7. The accompanying visualisation shows that your credit score increases with your annual income. Let's now investigate whether or not the monthly in-hand wage affects credit scores:

![newplot (2)](https://user-images.githubusercontent.com/118778677/224306514-63bc83c0-bb5d-4ecb-ae07-12a6ebff2e63.png)

8. Similar to annual income, your credit score will rise as you earn a higher monthly in-hand wage. So let's examine whether or not having multiple bank accounts affects credit scores:

![newplot (3)](https://user-images.githubusercontent.com/118778677/224306558-e1687dfd-1976-4239-8ff5-6286148d36c6.png)

9. Keeping more than five accounts open is bad for your credit score. A person should only have two or three bank accounts. Hence, having extra bank accounts has no beneficial effect on credit scores. Let's examine the effect that having a large number of credit cards has on your credit scores:

![newplot (4)](https://user-images.githubusercontent.com/118778677/224306668-9888156a-48de-48ae-a95d-f886175d6a98.png)

11. Having additional credit cards won't improve your credit scores, just like having more bank accounts won't. Your credit score will benefit from having 3 to 5 credit cards. Let's now examine how your average loan and EMI interest rate affects your credit scores:

![newplot (5)](https://user-images.githubusercontent.com/118778677/224307824-2f1f5333-d601-4323-bb63-b74a03659d71.png)

12. The credit score is good if the average interest rate ranges from 4 to 11%. Your credit scores will suffer if your average interest rate is more than 15%. Let's now look at how many loans you can take out at once with good credit:

![newplot (6)](https://user-images.githubusercontent.com/118778677/224307916-9b4fb1c3-59be-4392-a1a0-65ba6ad65bff.png)

13. You shouldn't take out more than one to three loans at once if you want to have decent credit. Your credit ratings will suffer if you have more than three loans open at once. Let's check your credit scores now to determine if missing a payment on time affects them negatively or not:

![newplot (7)](https://user-images.githubusercontent.com/118778677/224307974-d9daa538-fae4-4fdb-acf7-17b8b2c608c9.png)

14. You can postpone paying with a credit card 5 to 14 days beyond the due date. Your credit ratings may suffer if you make payments more than 17 days beyond the due date. Let's examine whether or not repeatedly missing payments will affect credit scores:

![newplot (8)](https://user-images.githubusercontent.com/118778677/224308049-509f642f-c307-401e-94a3-06e848028d71.png)

15. Delaying payments by 4 to 12 days from the due date won't lower your credit scores. Delaying payments by more than 12 days from the due date, however, will harm your credit scores. Let's now examine whether or not having greater debt will impact credit scores:

![newplot (9)](https://user-images.githubusercontent.com/118778677/224308135-2f8f71f9-887c-4810-ab33-398d60f6b201.png)

16. Your credit ratings won't be impacted by an unpaid balance of $380 to $1150. Yet having debt that is consistently more than $1338 will harm your credit scores. Let's examine whether or not having a high credit utilisation ratio will impact credit scores:

![newplot (10)](https://user-images.githubusercontent.com/118778677/224308306-25ecefda-50ee-4b8f-9a32-b07ce9e71634.png)

17. The term "credit usage ratio" refers to the percentage of your entire debt to your total credit line. The aforementioned statistic indicates that your credit ratings are unaffected by your credit use percentage. Let's now examine how a person's credit history age influences credit scores:

![newplot (11)](https://user-images.githubusercontent.com/118778677/224308374-42c3265d-5589-45a4-9263-12f40a38dd04.png)

18. Thus, having a long credit history raises credit ratings. Let's check how many EMIs a person with decent credit can have per month:

![newplot (12)](https://user-images.githubusercontent.com/118778677/224308458-fe6345fe-91ce-465b-8897-4ef25de3b919.png)

19. The quantity of EMIs you pay each month has little bearing on your credit scores. Let's check to see if your monthly investments have an impact on your credit scores:

![newplot (13)](https://user-images.githubusercontent.com/118778677/224308506-bb6cc064-7857-4eef-90b1-105ca75c0081.png)

20. Your credit ratings are not significantly impacted by the amount of money you invest each month. Now let's examine whether or not having a low balance at the end of the month has an impact on credit scores:

![newplot (15)](https://user-images.githubusercontent.com/118778677/224311961-8c4bc69f-bc68-4101-b26f-91ad7762af62.png)

So, having a large account balance at the end of each month is advantageous for your credit scores. Credit scores suffer with a balance of less than $250 each month.

Classification model Implementation 

Credit Mix, a further significant characteristic in the dataset, is useful for calculating credit scores. The credit mix feature details the various loans and credit products you have accessed.

I'll convert the categorical nature of the Credit Mix column into a numerical feature so that we can train a machine learning model to classify credit scores:

![image](https://user-images.githubusercontent.com/118778677/224310049-58218cf2-3171-458a-97fb-c5c6e70cc5a6.png)

1. I'll now separate the data into features and labels by choosing the features that we determined to be crucial to our model:

![image](https://user-images.githubusercontent.com/118778677/224309941-3eb94f5b-af62-41f6-bac5-a288ef4dcbaf.png)

2. As we move further, let's divide the data into training and test sets and train a credit score classification model:

![image](https://user-images.githubusercontent.com/118778677/224309857-2175dee0-834f-4f2d-9b28-310aa3bc0b66.png)

3. Let's Predict credit score classification model:

![image](https://user-images.githubusercontent.com/118778677/224309390-50424793-86e4-4a12-a74e-34cd33040004.png)

Hence, this is how Python users can utilise machine learning to classify credit scores.

Summary : Banks and credit card businesses can quickly make loans to borrowers with strong creditworthiness by categorising them based on their credit scores. Any bank or financial organisation will grant loans to someone with a good credit score.
