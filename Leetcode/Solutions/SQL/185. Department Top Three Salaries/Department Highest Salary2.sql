# Write your MySQL query statement below

select Department, Employee, Salary from (
    select 
    Department,
    Employee,
    a.Salary as Salary,
    @department,
    @rank:=if(
        @department=a.DepartmentId, 
        @rank+if(@salary=a.Salary,0,1),
        1
    ) as rank,
    @department:=a.DepartmentId,
    @salary:=a.Salary
    from (select @rank:=0, @department:=-1, @salary:=-1) c, 
    (
        select a.DepartmentId,
        b.Name as Department, a.Name as Employee,
        a.Salary 
        from Employee a
        join Department b on (a.DepartmentId = b.Id)
        order by a.DepartmentId, a.Salary desc
    )a
)a
where rank<=3
;
