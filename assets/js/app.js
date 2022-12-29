const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const sign_text = document.getElementById("sign_text")


const model =  tf.loadLayersModel('./model/model.json');

console.log("model", model)


var actions = ['sourd','rdv','covid','depression']
var sequence = []
var sentence = []
var predictions = []
var threshold = 0.5

var text_predictions = []


function onResults(results) {
    console.log("yup")

    keypoints = extract_keypoints(results)
    sequence.push(keypoints)

    sequence = sequence.slice(-30)


    if (sequence.length ==30){


        model.then(function (res) {
            prediction_tensor = res.predict(tf.tensor([sequence]));
            prediction = Array.from(prediction_tensor.dataSync())
            
            index = prediction.indexOf(Math.max(...prediction))

            
            console.log(predictions)
            console.log(actions[index])

            sign_text.innerHTML = ""
            sign_text.innerHTML = actions[index]
            
        }, function (err) {
            console.log(err);
        });

    }

    /*canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.segmentationMask, 0, 0,
        canvasElement.width, canvasElement.height);

    // Only overwrite existing pixels.
    canvasCtx.globalCompositeOperation = 'source-in';
    canvasCtx.fillStyle = '#00FF00';
    canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

    // Only overwrite missing pixels.
    canvasCtx.globalCompositeOperation = 'destination-atop';
    canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height);

    canvasCtx.globalCompositeOperation = 'source-over';
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
        { color: '#00FF00', lineWidth: 4 });
    drawLandmarks(canvasCtx, results.poseLandmarks,
        { color: '#FF0000', lineWidth: 2 });
    drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
        { color: '#C0C0C070', lineWidth: 1 });
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
        { color: '#CC0000', lineWidth: 5 });
    drawLandmarks(canvasCtx, results.leftHandLandmarks,
        { color: '#00FF00', lineWidth: 2 });
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
        { color: '#00CC00', lineWidth: 5 });
    drawLandmarks(canvasCtx, results.rightHandLandmarks,
        { color: '#FF0000', lineWidth: 2 });
    canvasCtx.restore();*/
}

function extract_keypoints(results){

    if (results.poseLandmarks){
        var pose = []
        for (var i =0;i<results.poseLandmarks.length;i++){
            pose.push(results.poseLandmarks[i]['x'])
            pose.push(results.poseLandmarks[i]['y'])
            pose.push(results.poseLandmarks[i]['z'])
            pose.push(results.poseLandmarks[i]['visibility'])
        }
    }
    else{
        var pose = new Array(33*4).fill(0)
    }
    if (results.leftHandLandmarks){
        var lh = []
        for (var i =0;i<results.leftHandLandmarks.length;i++){
            lh.push(results.leftHandLandmarks[i]['x'])
            lh.push(results.leftHandLandmarks[i]['y'])
            lh.push(results.leftHandLandmarks[i]['z'])
        }
    }
    else{
        var lh = new Array(21*3).fill(0)
    }
    if (results.rightHandLandmarks){
        var rh = []
        for (var i =0;i<results.rightHandLandmarks.length;i++){
            rh.push(results.rightHandLandmarks[i]['x'])
            rh.push(results.rightHandLandmarks[i]['y'])
            rh.push(results.rightHandLandmarks[i]['z'])
        }
    }
    else{
        var rh = new Array(21*3).fill(0)
    }
    return pose.concat(lh).concat(rh)
}



const holistic = new Holistic({
    locateFile: (file) => {
        return `./assets/js/holistic/${file}`;
        
    }
});
holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: true,
    smoothSegmentation: true,
    refineFaceLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});





holistic.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await holistic.send({ image: videoElement });
    },
    width: 500,
    height:300
});
camera.start();

