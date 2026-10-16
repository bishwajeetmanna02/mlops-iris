pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3.11 -m venv .venv
                    .venv/bin/python -m pip install --upgrade pip
                    .venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    .venv/bin/python train_model.py
                    test -f iris_model.pkl
                '''
            }
        }

        stage('Verify Flask App') {
            steps {
                sh '''
                    python3.11 -m py_compile app.py train_model.py
                    test -f iris_model.pkl
                '''
            }
        }
	stage('Deploy Model') {
 	   steps {
	        sh 'sudo /usr/local/bin/deploy-iris-model'
	   }
	}
    }

    post {
        success {
            echo 'Model training and application verification completed successfully.'
        }
        failure {
            echo 'Pipeline failed. Check the Console Output for the error.'
        }
    }
}
