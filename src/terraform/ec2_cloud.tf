# EC2-Instanz
resource "aws_instance" "cloud_instance" {
  ami                    = "ami-06b21ccaeff8cd686"
  instance_type          = "t2.micro"
  key_name               = "theObservatory"
  vpc_security_group_ids = [aws_security_group.theObservatory_sg.id]

  associate_public_ip_address = true

  user_data            = <<-EOF
  #!/bin/bash
  exec > /var/log/user-data.log 2>&1
  set -x

  sleep 10
  timedatectl set-timezone Europe/Vienna

  sudo su
  yum update -y
  yum install -y python3
  yum install -y python3-pip

  cd /home/ec2-user/
  pip install -r requirements.txt

  python server.py
EOF

  iam_instance_profile = "LabInstanceProfile"

  # First copy the application files
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

  # Then update the .env file
  provisioner "remote-exec" {
    inline = [
      "if grep -q '^NATS_URL=' /home/ec2-user/.env; then",
      "  sed -i 's|^NATS_URL=.*|NATS_URL=nats://${aws_instance.nats_instance.public_ip}:4222|' /home/ec2-user/.env",
      "else",
      "  echo 'NATS_URL=nats://${aws_instance.nats_instance.public_ip}:4222' >> /home/ec2-user/.env",
      "fi"
    ]

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
    volume_size           = 8
    volume_type           = "gp2"
    delete_on_termination = true
  }

  depends_on = [aws_instance.nats_instance]
}
