pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response time is less than 500ms", function () {
    pm.expect(pm.response.responseTime).to.be.below(500);
});

pm.test("Content-Type is present", function () {
    pm.response.to.have.header("Content-Type");
});



loc = pm.response.headers.get("Content-Type");
if(loc === 'application/json; charset=utf-8') {
// Access the response data JSON as a JavaScript object
var res = pm.response.json();
}

else if(loc === 'application/vnd.api+json; charset=utf-8') {
   var res = pm.response.json(); 
   res = res['data'];
}

else if(loc === 'application/ld+json; charset=utf-8') {
   var res = pm.response.json(); 
   res = res['hydra:member'];
}
else if(loc === 'application/hal+json; charset=utf-8') {
   var res = pm.response.json(); 
   res = res['_embedded']['item'];
}



else if(loc === 'application/xml; charset=utf-8' || loc === 'text/xml; charset=utf-8') {
   var res = xml2Json(responseBody);
    res = res['response']['item'];
   console.log(res);
}

else if (loc === 'text/csv; charset=utf-8') {



function convertCSVToJSON(str, delimiter = ',') {
    const titles = str.slice(0, str.indexOf('\n')).split(delimiter);
    const rows = str.slice(str.indexOf('\n') + 1).split('\n');
    return rows.map(row => {
        // Convert to 2D array
        const values = row.split(delimiter);
        // Convert array to object
        return titles.reduce((object, curr, i) => {
            object[curr] = values[i];
            return object;
        }, {})
    });
};



    var res = convertCSVToJSON(responseBody)
       console.log(res);
}

else {
   var res = pm.response.json(); 

   

}
console.log(res);


// -----------------------------
// - Structure data for charts -
// -----------------------------



   var capacity = [];
   var volume = [];
   var percentage = [];
   var validfrom = [];

   if(loc === 'application/vnd.api+json; charset=utf-8') {
   for (var i = 0, j = res.length; i < j; i++) {
       capacity.push(res[i]['attributes']['capacity']);
       volume.push(res[i]['attributes']['volume']);
       percentage.push(res[i]['attributes']['percentage']);
       validfrom.push(res[i]['attributes']['validfrom']);
    }
   }
   else {

   for (var i = 0, j = res.length; i < j; i++) {
       capacity.push(res[i]['capacity']);
       volume.push(res[i]['volume']);
       percentage.push(res[i]['percentage']);
       validfrom.push(res[i]['validfrom']);
    }
   }



// EDIT THIS OBJECT TO BIND YOUR DATA
const vizData = {
    
    // Labels take an array of strings

    // Data takes an array of numbers
    data: {

    //datasets[0].data = value.data

        datasets: [
            
                    {
            label: 'percentage',
            data: percentage,
            type: 'line',
            // this dataset is drawn on top
            yAxisID: 'y-axis-3',
            borderColor: "#bc5090",
            backgroundColor: "#bc5090",
            fill: 0,
            order: 0
        },
        {
            label: 'Capacity',
            data: capacity,
            type: 'line',
            // this dataset is drawn on top
            yAxisID: 'y-axis-2',
            borderColor: "#003f5c",
            backgroundColor: "#003f5c",
            fill: 0,
        },
            {
            label: 'Volume',
            data: volume,
            // this dataset is drawn below
            type: 'bar',
            backgroundColor: "#ffa600",
            fill: 1,
            yAxisID: 'y-axis-1',
        }, 

        ],
        labels: validfrom,
              
    }

};

// ------------
// - Template -
// ------------

// Configure the template
var template = `
<canvas id="myChart" height="120"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.5.0/Chart.min.js"></script> 

<script>
    // Get DOM element to render the chart in
    var ctx = document.getElementById("myChart");

    // Configure Chart JS here.
    // You can customize the options passed to this constructor to
    // make the chart look and behave the way you want
    var myChart = new Chart(ctx, {
        type: "bar",
        data: {
            datasets: [{
                data: [], // We will update this later in pm.getData()
                
                // Change these colours to customize the chart
                backgroundColor: ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],
            }]
        },
        options: {
             scales: {
            yAxes: [{
                type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
				display: true,
				position: 'left',
				id: 'y-axis-1',
                ticks: {
                    min: 0,
                    // Include a dollar sign in the ticks
                    callback: function(value, index, values) {
                        return formatkwh(value, true);
                    }
                }
            },
            {
                type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
				display: true,
				position: 'left',
				id: 'y-axis-2',
                ticks: {
                    min: 0,
                    // Include a dollar sign in the ticks
                    callback: function(value, index, values) {
                        return formatkwh(value, false);
                    }
                }
            },
                    {
                type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
				display: true,
				position: 'right',
				id: 'y-axis-3',
                ticks: {
                    min: 0,
                    // Include a dollar sign in the ticks
                    callback: function(value, index, values) {
                        return Math.round(value * 10) / 10 + ' %';
                    }
                }
            }]
        }
        }


    });

        function formatkwh(value, volume) {
                //Round to MW(h)
        value = Math.round(value / 1000) * 1000;
        //Prevent undefined return 0 W/Wh instead
        if (typeof value === 'undefined') {
            if (volume) {
                return 0 + " Wh";
            } else {
                return 0 + "W"
            }
        }
        value = value * 1000;
        var mod = 1000;
        if (volume) {
            var unitsText = 'Wh kWh MWh GWh TWh PWh';
        } else {
            var unitsText = 'W kW MW GW TW PW';
        }
        var units = unitsText.split(" ");
        for (i = 0; value > mod; i++) {
            value = value / mod;
        }

        
        //Round to 2 decimals
        value = Math.round(value * 100) / 100;
        //Convert number to sting with thousend seperators and . or , in the users/browser preference
        value = value.toLocaleString();
        //Atach unit to value
        value = value + ' ' + units[i];
        //Return value
        return  value;
    }

    // Access the data passed to pm.visualizer.set() from the JavaScript
    // code of the Visualizer template
    pm.getData(function (err, value) {
        console.log(value)
        myChart.data.datasets = value.data.datasets;
        myChart.data.labels = value.data.labels;
        myChart.update();
    });

</script>`;

// -------------------------
// - Bind data to template -
// -------------------------
        console.log(vizData)
// Set the visualizer template
pm.visualizer.set(template, vizData);
