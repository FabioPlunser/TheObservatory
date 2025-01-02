# EC2-Instanz
resource "aws_instance" "cloud_instance" {
  ami                   = "ami-06b21ccaeff8cd686"
  instance_type         = "t2.micro"
  key_name              = "theObservatory"
  vpc_security_group_ids = [aws_security_group.theObservatory_sg.id]


  associate_public_ip_address = true

  user_data = <<-EOF
  #!/bin/bash
  exec > /var/log/user-data.log 2>&1
  set -x

  sleep 10
  yum update -y
  #yum install -y docker
  yum install -y python3-pip python3-boto3 python3-psycopg2 python3-nats-py
  #service docker start
  #usermod -aG docker ec2-user
  timedatectl set-timezone Europe/Vienna
  #until sudo docker info; do sleep 5; done  # Warten, bis Docker verfügbar ist
EOF
  iam_instance_profile = "LabInstanceProfile"

  	  provisioner "file" {
	    source      = "../cloud/"
	    destination = "/home/ec2-user/"

	    connection {
	      type        = "ssh"
	      user        = "ec2-user"
	      private_key = file(var.private_pem_key)
	      host        = self.public_ip
	    }
	  }

  tags = {
    Name = "CloudInstance"
  }

  root_block_device {
    volume_size = 8
    volume_type = "gp2"
    delete_on_termination = true
  }
}

