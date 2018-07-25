
// let data;
// function loadJSON(file, callback){
// 	const xhr = new XMLHttpRequest();
// 	xhr.open('GET', 'data.json');


// 	xhr.onload = function() {
// 		console.log('here')

// 			callback(xhr.responseText);
		
// 	};
// 	xhr.onerror = (err) => console.log(err);

// 	xhr.send(null);
// }


// function load() {

// 	const messageElement = document.getElementById('message');
// 	messageElement.innerText = "check";
// 	loadJSON('data.json', function(response) {
// 		data = JSON.parse(response);
// 		console.log(data.name);
// 		messageElement.innerText = "inner";

// 	});

// }

// load();

// var fs = require('fs');
// var obj;
// fs.readFile('data.json', 'utf8', function(err, data) {
// 	if (err) throw err;
// 	obj = JSON.parse(data);
// 	console.log(obj)
// })

// http.createServer(function (req, res) {
//     res.writeHead(200, {'Content-Type': 'text/html'});
//     res.write('Hello');
//     res.end();
// }).listen(8080);




