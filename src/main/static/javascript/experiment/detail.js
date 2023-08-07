$(document).ready(function() {
    $( '#predict_form' ).each(function(){
        this.reset();
    });

    $('#upload_file').on('change', function () {
        //get the file name
        var fileName = $(this).val().replace("C:\\fakepath\\", "");

        //replace the "Choose a file" label
        $(this).next('.custom-file-label').html(fileName);
    });

    if (is_ready){
        draw_chart_feature_importance();
        draw_chart_performance();
        predict_selected_source();
        // draw_gauge_chart();
    }
});

function delete_confirm(delete_predict_id, delete_predict_name){
    $("#delete_predict_id").val(delete_predict_id)
    $("#delete_predict_name").text(delete_predict_name)
}

function delete_optimierung_confirm(delete_optimierung_id, delete_optimierung_name) {
    $("#delete_optimierung_id").val(delete_optimierung_id)
    $("#delete_optimierung_name").text(delete_optimierung_name)
}

function delete_gaugesetting_confirm(delete_gaugesetting_id) {
    $("#delete_gaugesetting_id").val(delete_gaugesetting_id)
}

function draw_chart_feature_importance(){

    var FImportanceContext = document.getElementById("chart_feature_importance").getContext("2d");
    var FImportanceData = {
        labels: feature_importance.label,
        datasets: [{
            label: "",
            data: feature_importance.data,
            backgroundColor: "#ade8f4",
            hoverBackgroundColor: "#f3722c"
        }]
    };
    var FImportanceChart = new Chart(FImportanceContext, {
        type: 'bar',
        data: FImportanceData,
        options: {
            scales: {
                xAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }],
                yAxes: [{
                    stacked: false,
                    ticks: {
                        beginAtZero: false
                    }
                }]
            },
            responsive: true,
            indexAxis: 'y',
            legend: {
                display: false
            },
        }
    });

}

function draw_chart_performance(){

    Highcharts.chart('chart_models', {

        chart: {
            type: 'bubble',
            plotBorderWidth: 1,
            zoomType: 'xy',
            height: 500
        },

        legend: {
            enabled: false
        },

        title: {
            text: ''
        },

        accessibility: {
            point: {
            valueDescriptionFormat: '{index}. {point.name}, Test score : {point.x}, Validation score: {point.y}'
            }
        },

        xAxis: {
            gridLineWidth: 1,
            title: {
                text: 'Validation score'
            },
            labels: {
                format: '{value}'
            },

            accessibility: {
                rangeDescription: 'Range: 60 to 100 grams.'
            }
        },
        credits: {
            enabled: false
        },
        yAxis: {
            startOnTick: false,
            endOnTick: false,
            title: {
                text: 'Test score'
            },
            labels: {
                format: '{value}'
            },
            maxPadding: 0.2,

            accessibility: {
                rangeDescription: 'Range: 0 to 160 grams.'
            }
        },

        tooltip: {
            useHTML: true,
            headerFormat: '<table>',
            pointFormat: '<tr><th colspan="2"><h4>{point.name}</h4></th></tr>' +
            '<tr><th>Validation :</th><td>{point.x}</td></tr>' +
            '<tr><th>Test:</th><td>{point.y}</td></tr>',
            footerFormat: '</table>',
            followPointer: true
        },

        plotOptions: {
            series: {
                dataLabels: {
                    enabled: true,
                        format: '{point.name}'
                }
            }
        },

        series: [{
            data: models_performance,
            color: 'rgba(173, 232, 244, 0.3)',
        }]
    });
}

// function draw_gauge_chart(){
//     var chartDom = document.getElementById('gaugeChart');
//     var myChart = echarts.init(chartDom);
//     var option;

//     option = {
//         series: [
//             {
//                 type: 'gauge',
//                 startAngle: 180,
//                 endAngle: 0,
//                 radius: '100%',
//                 min: lower_y.toFixed(1),
//                 max: upper_y.toFixed(1),
//                 axisLine: {
//                     lineStyle: {
//                         width: 30,
//                         color: [
//                             [(lower_bound - lower_y) / (upper_y - lower_y), '#fac858'],
//                             [(upper_bound-lower_y)/(upper_y-lower_y), '#37a2da'],
//                             [1, '#fd666d']
//                         ]
//                     }
//                 },
//                 pointer: {
//                     itemStyle: {
//                         color: 'auto'
//                     }
//                 },
//                 axisTick: {
//                     distance: -30,
//                     length: 8,
//                     lineStyle: {
//                         color: '#fff',
//                         width: 2
//                     }
//                 },
//                 splitLine: {
//                     distance: -30,
//                     length: 30,
//                     lineStyle: {
//                         color: '#fff',
//                         width: 4
//                     }
//                 },
//                 axisLabel: {
//                     color: 'auto',
//                     distance: 40,
//                     fontSize: 10,
//                 },
//                 detail: {
//                     valueAnimation: true,
//                     formatter: (value) => {
//                         return value.toFixed(2)
//                     },
//                     color: 'auto'
//                 },
//                 data: [
//                     {
//                         value: optimal_value,
//                         name: 'Optimal value'
//                     }
//                 ]
//             }
//         ],
//     };
//     option && myChart.setOption(option);
// }

function predict_selected_source() {
    var is_internal_selected = $("#radio_internal").is(":checked");

    if (is_internal_selected) {
        $("#select_internal_form").show();
        $("#select_external_form").hide();
    }
    else {
        $("#select_internal_form").hide();
        $("#select_external_form").show();
    }
}