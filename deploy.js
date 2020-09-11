//expect image as input
//array of 16 coordinates with probability
//works well at least 50% of the body is shown

let video
let poseNet
let pose
let skeleton

let brain
let poseLabel

// let state = 'waiting'
// let targetLabel

// function keyPressed(){
//     if (key =='s') {
//         brain.saveData()
//     } else {
//         targetLabel = key
//         console.log(targetLabel)
//         setTimeout(function (){
//             console.log('collecting')
//             state = 'collecting'   
    
//             setTimeout(function(){
//                 console.log('not collecting')
//                 state = 'waiting'
//             }, 10000)
    
//         }, 10000)
//     }
// }

function modelLoaded() {
    console.log('poseNet ready')
}

function setup(){
    createCanvas(640, 480)
    video = createCapture(VIDEO)
    video.hide()
    poseNet = ml5.poseNet(video, modelLoaded)
    poseNet.on('pose', gotPoses)

    let options = {
        inputs: 34,
        outputs: 4,
        task: 'classification',
        debug: true
    }
    brain = ml5.neuralNetwork(options)

    const modelInfo = {
        model: 'model.json',
        metadata: 'model_meta.json',
        weights: 'model.weights.bin'
    }
    brain.load(modelInfo, brainlLoaded)
}

function brainlLoaded(){
    console.log('pose classification ready!')

    classifyPose()
}

function classifyPose() {
    if (pose) {
        let inputs = []
        for (let i = 0; i<pose.keypoints.length; i++){
            let x = pose.keypoints[i].position.x
            let y = pose.keypoints[i].position.y
            inputs.push(x)
            inputs.push(y)
        }
        brain.classify(inputs, gotResult)
    } else {
        setTimeout(classifyPose, 100)
    }
}

function gotResult(error, results){
    if (results[0].confidence > 0.75) {
        poseLabel = results[0].label.toUpperCase();
    }
    console.log(results[0].confidence)
    // console.log(results)
    //console.log(results[0].label)
    classifyPose();
}

function gotPoses(poses){
    //onsole.log(poses)
    if(poses.length > 0){
        pose = poses[0].pose;
        skeleton = poses[0].skeleton
    }//can try addd confidence score into the neural network

}

function draw(){
    push()
    translate(video.width, 0)
    scale(-1, 1)
    image(video, 0, 0, video.width, video.height)

    if (pose) {
    for (let i = 0; i<pose.keypoints.length; i++){
        let x = pose.keypoints[i].position.x
        let y = pose.keypoints[i].position.y

        fill(0,255,0)
        ellipse(x,y,16,16)
    }

    for (let i = 0; i<skeleton.length; i++){
        let a = skeleton[i][0]
        let b = skeleton[i][1]
        strokeWeight(2)
        stroke(255)
        line(a.position.x, a.position.y, b.position.x, b.position.y)
    }

    }
    pop()

    fill(255, 0, 255)
    noStroke()
    textSize(512)
    textAlign(CENTER, CENTER)
    text(poseLabel, width / 2, height / 2)
}