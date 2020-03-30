# https://leetcode.com/problems/duplicate-emails/
# Write your MySQL query statement below
select distinct(p.Email) from Person p
where exists(select * from Person p1 where p1.id <> p.id and p.Email = p1.Email)
