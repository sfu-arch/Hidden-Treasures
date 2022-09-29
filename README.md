# Connecting to arch servers

# add this to the ~/.ssh/config
Host rcg
    HostName rcga-linux-ts1.dc.sfu.ca
    User <Your SFU computing ID>
    ServerAliveInterval 240
    
Host server-01 
    HostName  cs-arch-srv01.cmpt.sfu.ca
    # You can use cs-arch-srv02.cmpt.sfu.ca, cs-arch-srv03.cmpt.sfu.ca, cs-arch-srv04.cmpt.sfu.ca
    User <Your SFU computing ID>
    ProxyJump rcg
    ServerAliveInterval 240
    
/nfs : NFS mounted across all servers
/data: Local to each server.

Your password login will be same across all servers

## VSCoding to servers
[https://www.youtube.com/watch?v=-loT_5Gfi9Y](https://www.youtube.com/watch?v=-loT_5Gfi9Y)

## VPN-based access

If you have this you do not need ProxyJump access

[Step 0: Setup MFA.](https://www.sfu.ca/information-systems/services/mfa.html)
[Step 1: Set up vpn](https://www.sfu.ca/information-systems/services/sfu-vpn.html)
[Step 2: Connect VPN](https://www.sfu.ca/information-systems/services/sfu-vpn/how-to-guides/install-forticlient-vpn-app/windows.html)
[Step 3: SSH.]
