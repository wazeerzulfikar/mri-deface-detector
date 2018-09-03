var express = require('express');
app = express();
var path = require('path');

app.use(express.static('dist'));

app.get('/', function(request, response) {
  response.sendFile('dist/index.html');
});

app.listen(8080, () => console.log('Server Started'));
