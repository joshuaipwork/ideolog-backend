# Perfect Table Schema:

Bill Table:
- Bill ID ('id', str)
- Bill Title ('title', str)
- Bill Summary ('summary', str)
- Bill Short Summary ('short_summary', str)

Congressman Table:
- ID ('name', str)
- Congressman/Senator ('name', str)
- Chamber ('chamber', str)
- State ('state', str)
- Political Party..? ('party', str)
- Iteration (115 vs 116 Congress) ('iteration', int)

Vote History:
- Bill ID ('bill', str (foreign key to bill table))
- Congressman ('person', 'str' (foreign key to congressman table id field))
- Vote ('vote', int (0 for no, 1 for yes))