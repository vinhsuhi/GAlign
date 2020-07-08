    def train_embedding(self, GAlign, refinement_model, source_A_hat, target_A_hat, structural_optimizer):
        if self.args.noise_level == 0:
            print("Do not use augmentation")
            for epoch in range(self.args.emb_epochs):
                print("Structure learning epoch: {}".format(epoch))
                for i in range(2):
                    structural_optimizer.zero_grad()
                    if i == 0:
                        A_hat = source_A_hat
                        outputs = GAlign(source_A_hat, 's')
                    else:
                        A_hat = target_A_hat
                        outputs = GAlign(target_A_hat, 't')
                    loss = self.linkpred_loss(outputs[-1], A_hat)
                    if self.args.log:
                        print("Loss: {:.4f}".format(loss.data))
                    loss.backward()
                    structural_optimizer.step()
        else:
            print("Use Augmentation")
            new_source_A_hats = []
            new_target_A_hats = []
            new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'remove_edgse'))
            new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'add_edges'))
            new_source_A_hats.append(source_A_hat)
            new_source_feats = self.graph_augmentation(self.source_dataset, 'change_feats')
            new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'remove_edgse'))
            new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'add_edges'))
            new_target_A_hats.append(target_A_hat)
            new_target_feats = self.graph_augmentation(self.target_dataset, 'change_feats')

            GAlign.train()
            for epoch in range(self.args.emb_epochs):
                print("Structure learning epoch: {}".format(epoch))
                for i in range(2):
                    for j in range(len(new_source_A_hats)):
                        structural_optimizer.zero_grad()
                        if i == 0:
                            A_hat = source_A_hat
                            augment_A_hat = new_source_A_hats[j]
                            outputs = GAlign(source_A_hat, 's')
                            if j < 2:
                                augment_outputs = GAlign(augment_A_hat, 's')
                            else:
                                augment_outputs = GAlign(augment_A_hat, 's', new_source_feats)
                        else:
                            A_hat = target_A_hat
                            augment_A_hat = new_target_A_hats[j]
                            outputs = GAlign(target_A_hat, 't')
                            if j < 2:
                                augment_outputs = GAlign(augment_A_hat, 't')
                            else:
                                augment_outputs = GAlign(augment_A_hat, 't', new_target_feats)
                        structure_loss = self.linkpred_loss_multiple_layer(outputs, A_hat)
                        consistency_loss = self.get_consistency_loss(outputs, augment_outputs)
                        loss = (1-self.args.coe_consistency) * structure_loss + self.args.coe_consistency * consistency_loss
                        if self.args.log:
                            print("Loss: {:.4f}".format(loss.data))
                        loss.backward()
                        structural_optimizer.step()

        print("Done structural training")
        GAlign.eval()
        source_A_hat = source_A_hat.to_dense()
        target_A_hat = target_A_hat.to_dense()
        # refinement 
        if self.args.refine:
            S_max = None
            source_outputs = GAlign(refinement_model(source_A_hat, 's'), 's')
            target_outputs = GAlign(refinement_model(target_A_hat, 't'), 't')
            acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
            score = np.max(S, axis=1).mean()
            acc_max = 0
            alpha_source_max = None
            alpha_target_max = None
            if score > refinement_model.score_max:
                refinement_model.score_max = score
                alpha_source_max = refinement_model.alpha_source
                alpha_target_max = refinement_model.alpha_target
                acc_max = acc
                S_max = S
            print("Acc: {}, score: {:.4f}".format(acc, score))
                        
            for epoch in range(self.args.refinement_epochs):
                print("Refinement epoch: {}".format(epoch))
                source_candidates, target_candidates = self.get_candidate(source_outputs, target_outputs)
                
                refinement_model.alpha_source[source_candidates] *= 1.1
                refinement_model.alpha_target[target_candidates] *= 1.1
                source_outputs = GAlign(refinement_model(source_A_hat, 's'), 's')
                target_outputs = GAlign(refinement_model(target_A_hat, 't'), 't')
                acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
                score = np.max(S, axis=1).mean()
                if score > refinement_model.score_max:
                    refinement_model.score_max = score
                    alpha_source_max = refinement_model.alpha_source + 0
                    alpha_target_max = refinement_model.alpha_target + 0
                    acc_max = acc
                    S_max = S
                print("Acc: {}, score: {:.4f}, score_max {:.4f}".format(acc, score, refinement_model.score_max))
            print("Done refinement!")
            print("Acc with max score: {:.4f} is : {}".format(refinement_model.score_max, acc_max))
            refinement_model.alpha_source = alpha_source_max
            refinement_model.alpha_target = alpha_target_max
            self.S = S_max
            self.log_and_evaluate(GAlign, refinement_model, source_A_hat, target_A_hat)

        else:
            self.log_and_evaluate(GAlign, refinement_model, source_A_hat, target_A_hat)


