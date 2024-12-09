provider "aws" {
  region     = "us-east-1"
  access_key = "ASIAV3SHKFUBHAZQL7E2"
  secret_key = "OhqbbD29Pyr2hvGO/10WhLnBP3jVACTZVCbegBWT"
  token      = "IQoJb3JpZ2luX2VjELj//////////wEaCXVzLXdlc3QtMiJGMEQCIFgxgCreDPn/08tm0bkdiTA9b8RS0YBn2ZXXeY7avRNJAiBND7mSjlbfAx+ctiTQpFyYixw2V/QnQXCNetVLIXvqPSq1AghxEAIaDDQwMjgwMjc0ODY3NCIMMrrPsII21/I+OdPGKpICMpqGMZt4iZjt1mRFZstaXsIsu8jhcKdwDDlOo8BxRU1nWQZRfwfdlHS1+g5c85a6g5S9i8LXQ2w24VBofTSBkGP++vykxZvotgzPucCVTqaZi0bQmuUB2Ui3TBrMjEfgP2h07A6RxxI4ao45CDIoiJweyI2YXAmxlRjIHf64A23OiKqD9jEnFS9Qi/f3eC9LvPRnfpZH2U8yxEWuHUbvtbkZ5QP19WkJ4RrANBRFxYmudLQydUtTJ2wFH4KvzS/4twGoibPEZkU5eUtYrMbADtz0YCGGNTVOWWmKH03D8mIhKPCRHaBOq0Z5RKxmRTBq0vd0+H849wD54wfZ0p3+4k0FUKLhT28inMgC/NKcPJeG3DDJytq6BjqeAbUvyvWycRBnkIYMwKEInJH1Dbwi6uTdxeW/nsfQHrwYR0y+yI1uCvlVR2CXi+Szqqvi70/XNt/FQAqQYQJUxNrSoZn7yOgcmVWRBNvUU23ixLC/jhT6KMySQWxGSJKlnjjgVNtWWZ9g3zsCRIHv7wno2mWqWQHnhaEKrJq/Kzks8UuUHBR2Aojg3ww/HKa3CeL9vEPoKurUgYl7LXCG"
}

data "aws_iam_role" "lab_instance_role" {
  name = "LabRole"
}


resource "aws_security_group" "theObservatory_sg" {
  name_prefix = "theObservatory_sg"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
    ingress {
    from_port   = 4222
    to_port     = 4222
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
      ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}


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

