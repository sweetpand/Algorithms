CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
RETURN (
# Write your MySQL query statement below.
select distinct salary as getNthHighestSalary
from Employee E
where N-1=(select count(distinct salary) from Employee E2 where E2.Salary > E.Salary)

);
END
