# Write your MySQL query statement below
select s.id, visit_date, people
from stadium as s
where (
    s.people >= 100 and 
    (
       (select people from stadium where id = s.id-1) >= 100 and  
       (select people from stadium where id = s.id+1) >= 100  
        or 
       (select people from stadium where id = s.id-1) >= 100 and  
       (select people from stadium where id = s.id-2) >= 100 
        or 
       (select people from stadium where id = s.id+1) >= 100 and  
       (select people from stadium where id = s.id+2) >= 100  
    )
)
