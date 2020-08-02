# BankCustomerSegmentation

Note: It will be better to run this code in a python notebook instead of an editor such a PyCharm as it will ease the tasks and help in differentiating between them. 
As the code progresses and the number of plots increase, the difficulty of viewing each plot in an editor / IDE increases, so preferable use a Python Notebook such as Jupyter. 

In this project we have a bank's data set of it's customers for the past six months. The Data includes transaction frequency, amount, tenure, etc.

We have to divide the dataset into 4 groups wherein each group represents a set of customers that has similar features / properties. 

This analysis will help in customer segmentation or 'market segmentation' with the help of which the Bank can launch targeted marketing campaigns which are unique and specific to every customer group.   

The customer data has been provided in the for of a CSV file. 


                      THE REQUIRED CONDITIONS FOR CLUSTERING
First Customers cluster (Transactors): Those are customers who pay least amount of interest 
charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), 
Percentage of full payment = 23%

Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): 
highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance 
frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)

Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, 
target for increase credit limit and increase spending habits

Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance
