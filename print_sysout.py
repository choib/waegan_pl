import sys

def pr_status(args, epoch, length, i, g_loss, e_loss, wass_enc_loss, wass_loss, h_loss, time_left):
    if args.multi_critic:
        if args.clip_weight:
            str = "\rEpoch: [{}/{}],[Batch {}/{}] \tG Loss: {:<10.6e} \tE loss:{:<10.6e} \tD loss: {:<10.6e} \tF loss: {:<10.6e} \tH loss: {:<10.6e} \tETA {} ".format(
                epoch, 
                args.epochs, 
                i,
                length,
                g_loss.data.item(),
                e_loss.item(),
                wass_enc_loss.item(),
                wass_loss.data.item() if wass_loss != None else 0,
                h_loss.data.item() if h_loss != None else 0,
                time_left
            )
            #sys.stdout.write()
        else:
            str = "\rEpoch: [{}/{}],[Batch {}/{}] \tG Loss: {:<10.6e} \tE loss:{:<10.6e} \tD loss: {:<10.6e} \tF loss: {:<10.6e} \tH loss: {:<10.6e} \tgp loss: {:<10.6e} \tETA {} ".format(
                epoch, 
                args.epochs, 
                i,
                length,
                g_loss.data.item(),
                e_loss.item(),
                wass_enc_loss.item(),
                wass_loss.data.item() if wass_loss != None else 0,
                h_loss.item() if h_loss != None else 0,
                gp_loss.data.item() if gp_loss != None else 0,
                time_left
            )
            #sys.stdout.write(str)    
    else:
        str = "\rEpoch: [{}/{}],[Batch {}/{}] \tG Loss: {:<10.6e} \tE loss:{:<10.6e} \tD loss: {:<10.6e} \tETA {} ".format(
                epoch, 
                args.epochs, 
                i,
                length,
                g_loss.data.item(),
                e_loss.item(),
                wass_enc_loss.item(),
                #wass_loss.data.item(),
                #h_loss.data.item(),
                time_left
            )
    sys.stdout.write(str)
    return str
        