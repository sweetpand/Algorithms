A solution exists when:

There is no item of group B that appears between items of group A = No cycle between groups => We need a graph whose vertices are groups
Inside a group elements must be ordered = No cycle inside a group => We need a graph for each group
When the conditions above are verified, we can do topological sort to find a solution in two steps:

Topological sort between groups (so that group of items A appears before a group B)
Topological sort inside each group (so that an item a appears before an item b)
For the 2 graphs the edges are determined using the list beforeItems. beforeItems[i] = [j] gives:

if i and j don't belong to the same group : add an edge group[j] -> group[i] in the groups' graph
if i and j belong to the same group : add an edge j -> i in the graph of group group[i].
The cycle detection will be done when doing the topological sort.
