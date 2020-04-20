# Write your MySQL query statement below
SELECT     
        d.name      as "Department",                    
        e1.Name as "Employee",
        e1.Salary as "Salary"
       
    FROM 
        Employee e1 
        JOIN Employee e2  JOIN Department d
                      
    WHERE 
        e1.DepartmentId = e2.DepartmentId 
        AND e1.Salary <= e2.Salary  AND d.id = e2.DepartmentId
                            
                       
    GROUP BY d.name,e1.id
    HAVING COUNT(DISTINCT(e2.Salary)) <= 3
     order by d.name , salary desc  
