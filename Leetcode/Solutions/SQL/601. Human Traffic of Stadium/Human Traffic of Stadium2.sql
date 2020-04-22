select distinct s1.id, s1.visit_date, s1.people
from stadium s1, stadium s2, stadium s3
where s1.people >= 100 and
      s2.people >= 100 and
      s3.people >= 100 and
    ((s1.id+1 = s2.id and s1.id+2 = s3.id) or
     (s1.id-1 = s2.id and s1.id+1 = s3.id) or
     (s1.id-1 = s2.id and s1.id-2 = s3.id))
order by s1.id asc
