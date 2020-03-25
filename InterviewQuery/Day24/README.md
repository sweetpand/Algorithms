Interview Query Question #24 | Rolling Bank Transactions

This question was asked by: Dropbox
`bank_transactions` table

column	type
user_id	int
created_at	datetime
transaction_value	float
We're given a table bank transactions with three columns, user_id, a deposit or withdrawal value, and created_at time for each transaction.

Write a query to get the total three day rolling average for deposits by day.

Example:

Input

user_id	created_at	transaction_value
1	2019-01-01	10
2	2019-01-01	20
1	2019-01-02	-10
1	2019-01-02	50
2	2019-01-03	5
3	2019-01-03	5
2	2019-01-04	10
1	2019-01-04	10
Output

dt	rolling_three_day
2019-01-01	30.00
2019-01-02	40.00
2019-01-03	30.00
2019-01-04	23.33
