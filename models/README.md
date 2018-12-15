**[VGG19](https://github.com/Quan-Sun/Dive-into-Machine-Learning/blob/master/models/VGG19.ipynb)** - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

<div align=center><img src="https://github.com/Quan-Sun/Dive-into-Machine-Learning/blob/master/models/images/VGG19.png"/></div>

**[ResNet50](https://github.com/Quan-Sun/Dive-into-Machine-Learning/blob/master/models/RestNet50.ipynb)** - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

<html>
<head>
    <meta charset="utf-8">
    <title>Netscope</title>
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" href="assets/css/tooltip.css">
    <link rel="stylesheet" href="assets/css/codemirror.css">
    <link rel="stylesheet" href="assets/css/netscope.css">
    <script src="assets/js/lib/lodash.min.js"></script>
    <script src="assets/js/lib/jquery.min.js"></script>
    <script src="assets/js/lib/jquery.qtip.min.js"></script>
    <script src="assets/js/lib/d3.min.js"></script>
    <script src="assets/js/lib/dagre-d3.min.js"></script>
    <script src="assets/js/lib/director.min.js"></script>
    <script src="assets/js/netscope.js"></script>
</head>
<body>
<div id="master-container">
    <div id='net-column' class="column">
        <div id="net-spinner">
            <img src="assets/img/loading.svg" title="Loading network..." />
        </div>
        <div id="net-error" style="display:none">
            <div class="title">Error Encountered</div>
            <div class="msg"></div>
        </div>
        <div id="net-warning" style="display:none">
            <div class="title">Warning</div>
            <div class="msg"></div>
        </div>
        <div id="net-container">
            <div id="net-group">
                <h1 id="net-title"></h1>
                <svg id="net-svg"></svg>
            </div>
        </div>
    </div>
</div>
</body>
</html>
