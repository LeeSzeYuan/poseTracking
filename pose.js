//expect image as input
//array of 16 coordinates with probability
//works well at least 50% of the body is shown

let video
let poseNet
let pose;
let skeleton

let brain;

let state = 'waiting'
let targetLabel

function keyPressed(){
    targetLabel = key
    console.log(targetLabel)
    setTimeout(function (){
        console.log('collecting')
        state = 'collecting'
    }, 1000)
    
}

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
}

function gotPoses(poses){
    console.log(poses)
    if(poses.length > 0){
        pose = poses[0].pose;
        skeleton = poses[0].skeleton
    }
}

function draw(){
    translate(video.width, 0)
    scale(-1, 1)
    image(video, 0, 0, video.width, video.height)

    if (pose) {
        let eyeR = pose.rightEye
        let eyeL = pose.leftEye
        let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y)

        fill(255, 0, 0)
        ellipse(pose.nose.x, pose.nose.y, d/2)
        fill(0,0, 255)
        ellipse(pose.leftWrist.x, pose.leftWrist.y, 32)
        ellipse(pose.rightWrist.x, pose.rightWrist.y, 32)
    
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
}
