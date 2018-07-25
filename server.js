const express = require('express')
app = express()
var path = require('path')

app.use(express.static('public'));

app.get('/', function(request, response) {
	response.sendFile('public/index.html');

});

app.listen(8080, ()=>console.log('Server Started'));

