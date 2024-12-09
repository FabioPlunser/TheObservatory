# EC2-Instanz
resource "aws_instance" "connect4_instance" {
  ami                   = "ami-06b21ccaeff8cd686"
  instance_type         = "t2.micro"
  key_name              = "connect4"
  vpc_security_group_ids = [aws_security_group.theObservatory_sg.id]


  associate_public_ip_address = true

  user_data = <<-EOF
  #!/bin/bash
  exec > /var/log/user-data.log 2>&1
  set -x

  sleep 10
  yum update -y
  yum install -y docker
  service docker start
  usermod -aG docker ec2-user
  timedatectl set-timezone Europe/Vienna
  until sudo docker info; do sleep 5; done  # Warten, bis Docker verfügbar ist
  sudo docker run -d --name nats -p 4222:4222 -e NATS_URI=nats://nats:4222 nats:latest
EOF
  iam_instance_profile = "LabInstanceProfile"

  tags = {
    Name = "NatsInstance"
  }

  root_block_device {
    volume_size = 8
    volume_type = "gp2"
    delete_on_termination = true
  }
}

