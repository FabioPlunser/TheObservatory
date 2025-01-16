# EC2-Instanz
resource "aws_instance" "cloud_instance" {
  ami                    = "ami-06b21ccaeff8cd686"
  instance_type          = "t2.micro"
  key_name               = "theObservatory"
  vpc_security_group_ids = [aws_security_group.theObservatory_sg.id]

  associate_public_ip_address = true

  user_data = <<-EOF
  #!/bin/bash
  exec > /var/log/user-data.log 2>&1
  set -x

  # System setup only
  yum update -y
  yum install -y python3 python3-pip aws-cli nc python3-devel gcc
  timedatectl set-timezone Europe/Vienna
EOF

  # Use the pre-existing LabRole instead of creating a new one
  iam_instance_profile = "LabInstanceProfile"

  # First copy application files
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

  # Then initialize and configure the server
provisioner "remote-exec" {
    inline = [
      "ls -la /home/ec2-user/",
      "python3 -m venv /home/ec2-user/venv",
      "source /home/ec2-user/venv/bin/activate",
      "pip3 install --upgrade pip wheel setuptools",
      "pip3 install -r /home/ec2-user/requirements.txt",
      "echo 'NATS_URL=nats://${aws_instance.nats_instance.public_ip}:4222' > /home/ec2-user/.env",
      "echo 'REGION=${var.region}' >> /home/ec2-user/.env",
      "echo 'BUCKET_NAME=${var.bucket_name}' >> /home/ec2-user/.env", 
      "sudo chown -R ec2-user:ec2-user /home/ec2-user/",
      "sudo chmod 600 /home/ec2-user/.env",
      "sudo bash -c 'cat > /etc/systemd/system/cloud-server.service << EOL\n[Unit]\nDescription=Cloud Server\nAfter=network.target\n\n[Service]\nType=simple\nUser=ec2-user\nWorkingDirectory=/home/ec2-user\nEnvironmentFile=/home/ec2-user/.env\nEnvironment=PATH=/home/ec2-user/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin\nExecStart=/home/ec2-user/venv/bin/python3 /home/ec2-user/server.py\nRestart=always\nRestartSec=3\n\n[Install]\nWantedBy=multi-user.target\nEOL'",
      "sudo systemctl daemon-reload",
      "sudo systemctl enable cloud-server",
      "sudo systemctl start cloud-server",
      "sleep 5",
      "sudo systemctl status cloud-server --no-pager || true",
      "sudo journalctl -u cloud-server --no-pager -n 20 || true"
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
